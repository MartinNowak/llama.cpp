#pragma once

// Residency of a paged KV buffer over independently backed units.
//
// The buffer is one virtual reservation cut into units of the driver's
// minimum allocation granularity, and every unit is backed by its own
// physical handle, its own mapping, and its own access grant. This engine
// owns the unit table and the five accounting quantities and performs the
// two transactions a residency policy consists of: commit, which backs every
// required unit that is unbacked, and reclaim, which returns every backed
// unit the requirement set has released. It calls the driver through a table
// of function pointers rather than through the CUDA driver API, so the
// transaction ordering and the unwind on a failed step are exercised by a
// host-only test with a mocked driver, and the CUDA backend supplies the
// cuMem* implementations at run time.
//
// Commit order per unit: create, map, grant access, zero, and only then does
// the unit count as mapped and readable. A failure at any step unwinds every
// unit this transaction created, in reverse, and leaves the units that were
// backed before it untouched. Reclaim order: quiesce the device once, then
// unmap and release each candidate. An unmapped unit is released outright,
// so physical_retained_unmapped_bytes reads zero while every release
// succeeds; a unit whose release the driver refused keeps its handle, counts
// as retained unmapped until the next commit of that unit retries the
// release, and is never overwritten by a new allocation. A released unit's
// bytes accumulate in physical_released_bytes.

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_paged_kv_driver {
    void * user = nullptr;
    // Every operation returns 0 on success and a driver status otherwise.
    int (*create)(void * user, size_t unit_index, uint64_t * handle) = nullptr;
    int (*map)(void * user, size_t unit_index, uint64_t handle) = nullptr;
    int (*set_access)(void * user, size_t unit_index) = nullptr;
    // Zero the unit and complete the write on the host before returning.
    int (*zero)(void * user, size_t unit_index) = nullptr;
    // Retire every outstanding device operation that could reference the buffer.
    int (*quiesce)(void * user) = nullptr;
    int (*unmap)(void * user, size_t unit_index) = nullptr;
    int (*release)(void * user, uint64_t handle) = nullptr;
};

struct ggml_paged_kv_unit {
    uint64_t handle = 0;
    bool created = false;
    bool mapped = false;
    bool published = false; // access granted and zeroed: readable by the device
    bool retained = false;  // unmapped, and its release refused: held, counted as retained unmapped
};

struct ggml_paged_kv_accounting {
    size_t virtual_reserved_bytes = 0;
    size_t physical_allocated_bytes = 0;
    size_t physical_mapped_bytes = 0;
    size_t physical_retained_unmapped_bytes = 0;
    size_t physical_released_bytes = 0;
};

struct ggml_paged_kv_transaction {
    bool ok = true;
    const char * step = "none";   // the driver step that failed
    size_t unit_index = 0;        // the unit it failed on
    int status = 0;               // the driver status it returned
    size_t units_committed = 0;   // units published by this transaction
    size_t units_released = 0;    // units released by this transaction
    size_t units_unwound = 0;     // units created by a failed commit and released again
    size_t units_unwind_failed = 0; // units a failed commit created that its unwind could not release
};

class ggml_paged_kv_residency {
public:
    ggml_paged_kv_residency(size_t unit_bytes, size_t virtual_reserved_bytes)
        : unit_bytes_(unit_bytes) {
        accounting_.virtual_reserved_bytes = virtual_reserved_bytes;
        units_.resize(unit_bytes > 0 ? virtual_reserved_bytes / unit_bytes : 0);
    }

    size_t unit_bytes() const { return unit_bytes_; }
    size_t unit_count() const { return units_.size(); }
    const ggml_paged_kv_accounting & accounting() const { return accounting_; }
    bool unit_published(size_t i) const { return i < units_.size() && units_[i].published; }

    size_t published_count() const {
        size_t n = 0;
        for (const auto & u : units_) {
            n += u.published ? 1 : 0;
        }
        return n;
    }

    // units([a, b)) = {floor(a/G), ..., floor((b-1)/G)}; an empty interval names no unit.
    static bool units_of(size_t unit_bytes, size_t offset, size_t size, size_t & first, size_t & last_p1) {
        if (size == 0 || unit_bytes == 0) {
            first = 0;
            last_p1 = 0;
            return false;
        }
        first = offset / unit_bytes;
        last_p1 = (offset + size - 1) / unit_bytes + 1;
        return true;
    }

    // Mark the units of one byte interval required. The interval has to lie
    // inside the reservation; the caller computed it from a tensor the
    // allocator placed there.
    bool require(std::vector<bool> & required, size_t offset, size_t size) const {
        if (required.size() != units_.size()) {
            required.assign(units_.size(), false);
        }
        if (size == 0) {
            return true;
        }
        if (offset > accounting_.virtual_reserved_bytes || size > accounting_.virtual_reserved_bytes - offset) {
            return false;
        }
        size_t first = 0;
        size_t last_p1 = 0;
        units_of(unit_bytes_, offset, size, first, last_p1);
        for (size_t i = first; i < last_p1; ++i) {
            required[i] = true;
        }
        return true;
    }

    // Whether every unit of the interval is published.
    bool resident(size_t offset, size_t size) const {
        if (size == 0) {
            return true;
        }
        if (offset > accounting_.virtual_reserved_bytes || size > accounting_.virtual_reserved_bytes - offset) {
            return false;
        }
        size_t first = 0;
        size_t last_p1 = 0;
        units_of(unit_bytes_, offset, size, first, last_p1);
        for (size_t i = first; i < last_p1; ++i) {
            if (!units_[i].published) {
                return false;
            }
        }
        return true;
    }

    // Call fn(offset, size) for every maximal run of published bytes inside
    // the interval, so a memset or copy touches resident storage alone.
    template <typename F>
    void for_each_resident_run(size_t offset, size_t size, F fn) const {
        if (size == 0) {
            return;
        }
        const size_t end = offset + size;
        size_t first = 0;
        size_t last_p1 = 0;
        units_of(unit_bytes_, offset, size, first, last_p1);
        size_t run_start = 0;
        bool in_run = false;
        for (size_t i = first; i <= last_p1; ++i) {
            const bool published = i < last_p1 && units_[i].published;
            if (published && !in_run) {
                run_start = i;
                in_run = true;
            } else if (!published && in_run) {
                const size_t a = run_start * unit_bytes_ > offset ? run_start * unit_bytes_ : offset;
                const size_t b = i * unit_bytes_ < end ? i * unit_bytes_ : end;
                fn(a, b - a);
                in_run = false;
            }
        }
    }

    // Back every required unit that is unpublished. Transactional: a failed
    // step unwinds the units this call created and leaves the rest as they were.
    ggml_paged_kv_transaction commit(const std::vector<bool> & required, const ggml_paged_kv_driver & drv) {
        ggml_paged_kv_transaction tx;
        std::vector<size_t> created;
        for (size_t i = 0; i < units_.size() && i < required.size(); ++i) {
            if (!required[i] || units_[i].published) {
                continue;
            }
            ggml_paged_kv_unit & u = units_[i];
            // A unit whose earlier release was refused still holds its
            // handle; that release is retried ahead of any new allocation,
            // so a handle is never overwritten while the driver holds it.
            if (u.created) {
                ggml_paged_kv_transaction retry;
                if (!release_unit(i, drv, retry)) {
                    return unwind(tx, retry.step, i, retry.status, created, drv);
                }
            }
            int status = drv.create(drv.user, i, &u.handle);
            if (status != 0) {
                u.handle = 0;
                return unwind(tx, "create", i, status, created, drv);
            }
            u.created = true;
            created.push_back(i);
            accounting_.physical_allocated_bytes += unit_bytes_;
            status = drv.map(drv.user, i, u.handle);
            if (status != 0) {
                return unwind(tx, "map", i, status, created, drv);
            }
            u.mapped = true;
            status = drv.set_access(drv.user, i);
            if (status != 0) {
                return unwind(tx, "set_access", i, status, created, drv);
            }
            status = drv.zero(drv.user, i);
            if (status != 0) {
                return unwind(tx, "zero", i, status, created, drv);
            }
            u.published = true;
            accounting_.physical_mapped_bytes += unit_bytes_;
            tx.units_committed += 1;
        }
        return tx;
    }

    // Return every published unit the requirement set has released. The
    // device is quiesced once ahead of the first unmap, and only where there
    // is a candidate, so a call that releases nothing costs no synchronization.
    ggml_paged_kv_transaction reclaim(const std::vector<bool> & required, const ggml_paged_kv_driver & drv) {
        ggml_paged_kv_transaction tx;
        std::vector<size_t> candidates;
        for (size_t i = 0; i < units_.size(); ++i) {
            const bool needed = i < required.size() && required[i];
            if (units_[i].published && !needed) {
                candidates.push_back(i);
            }
        }
        if (candidates.empty()) {
            return tx;
        }
        const int status = drv.quiesce(drv.user);
        if (status != 0) {
            tx.ok = false;
            tx.step = "quiesce";
            tx.status = status;
            return tx;
        }
        for (size_t i : candidates) {
            if (!release_unit(i, drv, tx)) {
                return tx;
            }
            tx.units_released += 1;
        }
        return tx;
    }

    // Unmap and release every unit; the reservation is the caller's to free.
    ggml_paged_kv_transaction destroy(const ggml_paged_kv_driver & drv) {
        ggml_paged_kv_transaction tx;
        for (size_t i = 0; i < units_.size(); ++i) {
            if (units_[i].created) {
                release_unit(i, drv, tx);
            }
        }
        return tx;
    }

private:
    // Unmap where mapped and release where created, adjusting the accounting
    // per step, so a unit half-committed by a failed transaction is unwound
    // by the same path a reclaim takes.
    bool release_unit(size_t i, const ggml_paged_kv_driver & drv, ggml_paged_kv_transaction & tx) {
        ggml_paged_kv_unit & u = units_[i];
        if (u.mapped) {
            const int status = drv.unmap(drv.user, i);
            if (status != 0) {
                tx.ok = false;
                tx.step = "unmap";
                tx.unit_index = i;
                tx.status = status;
                return false;
            }
            if (u.published) {
                accounting_.physical_mapped_bytes -= unit_bytes_;
            }
            u.mapped = false;
            u.published = false;
        }
        if (u.created) {
            const int status = drv.release(drv.user, u.handle);
            if (status != 0) {
                // The handle stays with the unit and its bytes move from
                // mapped to retained unmapped: held by the driver, readable
                // by nothing, and retried by the next commit of the unit.
                if (!u.retained) {
                    u.retained = true;
                    accounting_.physical_retained_unmapped_bytes += unit_bytes_;
                }
                tx.ok = false;
                tx.step = "release";
                tx.unit_index = i;
                tx.status = status;
                return false;
            }
            if (u.retained) {
                u.retained = false;
                accounting_.physical_retained_unmapped_bytes -= unit_bytes_;
            }
            accounting_.physical_allocated_bytes -= unit_bytes_;
            accounting_.physical_released_bytes += unit_bytes_;
            u.created = false;
            u.handle = 0;
        }
        return true;
    }

    ggml_paged_kv_transaction unwind(ggml_paged_kv_transaction tx, const char * step, size_t unit_index, int status,
                                     const std::vector<size_t> & created, const ggml_paged_kv_driver & drv) {
        tx.ok = false;
        tx.step = step;
        tx.unit_index = unit_index;
        tx.status = status;
        // Units published by this transaction count against it too: the
        // caller sees the state it started from.
        tx.units_committed = 0;
        for (auto it = created.rbegin(); it != created.rend(); ++it) {
            ggml_paged_kv_transaction unwind_tx;
            if (release_unit(*it, drv, unwind_tx)) {
                tx.units_unwound += 1;
            } else {
                tx.units_unwind_failed += 1;
            }
        }
        return tx;
    }

    size_t unit_bytes_ = 0;
    std::vector<ggml_paged_kv_unit> units_;
    ggml_paged_kv_accounting accounting_;
};
