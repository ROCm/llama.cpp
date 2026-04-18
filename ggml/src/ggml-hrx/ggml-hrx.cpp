#include "ggml-hrx.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "kernels/hrx_kernel_catalog.h"
#include "hrx_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;
static constexpr size_t GGML_HRX_STAGING_ARENA_DEFAULT_SIZE = 8 * 1024 * 1024;

struct ggml_backend_hrx_staging_arena {
    hrx_stream_t stream = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * mapped = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

enum class ggml_backend_hrx_provider_kind {
    none,
    hsaco,
};

struct ggml_backend_hrx_op_provider {
    ggml_backend_hrx_provider_kind kind = ggml_backend_hrx_provider_kind::none;
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};

    ggml_backend_hrx_op_provider() = default;
    ggml_backend_hrx_op_provider(const ggml_backend_hrx_op_provider &) = delete;
    ggml_backend_hrx_op_provider & operator=(const ggml_backend_hrx_op_provider &) = delete;

    void reset() {
        if (executable) {
            hrx_executable_release(executable);
        }
        kind = ggml_backend_hrx_provider_kind::none;
        executable = nullptr;
        export_ordinal = 0;
        export_info = {};
    }

    ~ggml_backend_hrx_op_provider() {
        reset();
    }
};

struct ggml_backend_hrx_elementwise_constants {
    int64_t n;
};

static_assert(sizeof(ggml_backend_hrx_elementwise_constants) == 8);

struct ggml_backend_hrx_device_context {
    hrx_device_t device = nullptr;
    hrx_stream_t active_stream = nullptr;
    hrx_stream_t transfer_stream = nullptr;
    std::mutex streams_mutex;
    std::vector<hrx_stream_t> live_streams;
    std::vector<ggml_backend_hrx_staging_arena> staging_arenas;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    ggml_backend_hrx_op_provider add_provider;
};

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream);

struct ggml_backend_hrx_reg_context {
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context() {
        for (auto & device_context : device_contexts) {
            if (device_context) {
                device_context->add_provider.reset();
            }
            if (device_context && device_context->transfer_stream) {
                hrx_status_t status = hrx_stream_synchronize(device_context->transfer_stream);
                if (!hrx_status_is_ok(status)) {
                    hrx_status_ignore(status);
                }
                ggml_backend_hrx_unregister_stream(device_context.get(), device_context->transfer_stream);
                hrx_stream_release(device_context->transfer_stream);
                device_context->transfer_stream = nullptr;
            }
            if (device_context && device_context->device) {
                hrx_device_release(device_context->device);
                device_context->device = nullptr;
            }
        }
        if (gpu_initialized) {
            hrx_status_t status = hrx_gpu_shutdown();
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
        }
    }
};

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    std::string name;
    hrx_buffer_params_t params = {};
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
};

static bool ggml_backend_hrx_log_status(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX_CHECK(expr) ggml_backend_hrx_log_status((expr), #expr, __FILE__, __LINE__)

static uint64_t ggml_backend_hrx_u64_from_env(const char * name, uint64_t default_value) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        GGML_LOG_WARN("%s: ignoring invalid %s=%s\n", __func__, name, value);
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

static size_t ggml_backend_hrx_align_up(size_t value, size_t alignment) {
    GGML_ASSERT(alignment > 0);
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static size_t ggml_backend_hrx_staging_arena_capacity() {
    const uint64_t requested = ggml_backend_hrx_u64_from_env(
        "GGML_HRX_STAGING_ARENA_SIZE", GGML_HRX_STAGING_ARENA_DEFAULT_SIZE);
    const size_t capacity = static_cast<size_t>(std::max<uint64_t>(requested, GGML_HRX_ALIGNMENT));
    return ggml_backend_hrx_align_up(capacity, GGML_HRX_ALIGNMENT);
}

static ggml_guid_t ggml_backend_hrx_guid(void) {
    static ggml_guid guid = { 0x1c, 0x65, 0x79, 0x0a, 0x31, 0x8b, 0x4d, 0xa6, 0x9e, 0x16, 0x6f, 0x13, 0x39, 0xb2, 0xe7, 0x5c };
    return &guid;
}

static ggml_backend_hrx_device_context * ggml_backend_hrx_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx_device_context *>(dev->context);
}

static ggml_backend_hrx_buffer_type_context * ggml_backend_hrx_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static bool ggml_backend_hrx_tensor_buffer_ref(
        const ggml_tensor * tensor, hrx_buffer_ref_t * out_ref) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (!context->buffer) {
        return false;
    }

    const size_t offset = ggml_backend_hrx_tensor_offset(context, tensor);
    const size_t length = ggml_nbytes(tensor);
    if (offset > buffer->size || length > buffer->size - offset) {
        GGML_LOG_ERROR(
            "%s: tensor %s has out-of-bounds HRX buffer ref: offset=%zu length=%zu buffer_size=%zu\n",
            __func__, tensor->name, offset, length, buffer->size);
        return false;
    }

    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ offset,
        /* .length = */ length,
    };
    return true;
}

static void ggml_backend_hrx_register_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    if (std::find(device_context->live_streams.begin(), device_context->live_streams.end(), stream) ==
            device_context->live_streams.end()) {
        device_context->live_streams.push_back(stream);
    }
}

static void ggml_backend_hrx_reset_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena.retired_buffers.clear();
    arena.offset = 0;
}

static void ggml_backend_hrx_release_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    if (arena.buffer) {
        hrx_buffer_release(arena.buffer);
    }
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena = {};
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_find_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    for (auto & arena : device_context->staging_arenas) {
        if (arena.stream == stream) {
            return &arena;
        }
    }
    return nullptr;
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_get_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
        return arena;
    }
    device_context->staging_arenas.push_back({});
    auto & arena = device_context->staging_arenas.back();
    arena.stream = stream;
    return &arena;
}

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    auto & streams = device_context->live_streams;
    streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
    auto & arenas = device_context->staging_arenas;
    auto arena_it = std::find_if(
        arenas.begin(), arenas.end(),
        [stream](const ggml_backend_hrx_staging_arena & arena) { return arena.stream == stream; });
    if (arena_it != arenas.end()) {
        ggml_backend_hrx_release_staging_arena_locked(*arena_it);
        arenas.erase(arena_it);
    }
    if (device_context->active_stream == stream) {
        device_context->active_stream = nullptr;
    }
}

static hrx_stream_t ggml_backend_hrx_retain_timeline_stream(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t stream = device_context->active_stream;
    if (!stream) {
        stream = device_context->transfer_stream;
    }
    if (stream) {
        hrx_stream_retain(stream);
    }
    return stream;
}

static bool ggml_backend_hrx_sync_streams(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    bool ok = true;
    for (hrx_stream_t stream : device_context->live_streams) {
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_prepare_stream_signal(
        hrx_stream_t stream,
        hrx_semaphore_t * semaphore,
        uint64_t * signal_value,
        hrx_semaphore_list_t * wait_list,
        hrx_semaphore_list_t * signal_list,
        hrx_semaphore_t * wait_semaphores,
        uint64_t * wait_values,
        hrx_semaphore_t * signal_semaphores,
        uint64_t * signal_values) {
    hrx_timeline_point_t position = {};
    if (!GGML_HRX_CHECK(hrx_stream_flush(stream)) ||
        !GGML_HRX_CHECK(hrx_stream_get_timeline_position(stream, &position)) ||
        !GGML_HRX_CHECK(hrx_stream_get_semaphore(stream, semaphore))) {
        return false;
    }

    *signal_value = position.value + 1;
    if (position.value > 0) {
        wait_semaphores[0] = *semaphore;
        wait_values[0] = position.value;
        *wait_list = {
            /* .semaphores = */ wait_semaphores,
            /* .values     = */ wait_values,
            /* .count      = */ 1,
        };
    } else {
        *wait_list = {};
    }

    signal_semaphores[0] = *semaphore;
    signal_values[0] = *signal_value;
    *signal_list = {
        /* .semaphores = */ signal_semaphores,
        /* .values     = */ signal_values,
        /* .count      = */ 1,
    };
    return true;
}

static bool ggml_backend_hrx_finish_stream_signal(hrx_stream_t stream, uint64_t signal_value) {
    uint64_t advanced_value = 0;
    if (!GGML_HRX_CHECK(hrx_stream_advance_timeline(stream, &advanced_value))) {
        return false;
    }
    if (advanced_value != signal_value) {
        GGML_LOG_ERROR("%s: stream timeline advanced to %" PRIu64 ", expected %" PRIu64 "\n",
                __func__, advanced_value, signal_value);
        return false;
    }
    return GGML_HRX_CHECK(hrx_stream_wait(stream));
}

static bool ggml_backend_hrx_queue_fill_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t buffer,
        size_t offset,
        size_t size,
        const void * pattern,
        size_t pattern_size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous fill\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_fill(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, buffer, offset, size, pattern, pattern_size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_queue_copy_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t src,
        size_t src_offset,
        hrx_buffer_t dst,
        size_t dst_offset,
        size_t size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous copy\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_copy(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, src, src_offset, dst, dst_offset, size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_ensure_staging_buffer_locked(
        ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_staging_arena * arena,
        size_t required_capacity) {
    if (arena->buffer && arena->capacity >= required_capacity && arena->mapped) {
        return true;
    }

    if (arena->buffer) {
        arena->retired_buffers.push_back(arena->buffer);
        arena->buffer = nullptr;
        arena->mapped = nullptr;
        arena->capacity = 0;
        arena->offset = 0;
    }

    const size_t capacity = ggml_backend_hrx_align_up(
        std::max(required_capacity, ggml_backend_hrx_staging_arena_capacity()),
        GGML_HRX_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT |
                       HRX_BUFFER_USAGE_MAPPING_SCOPED |
                       HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &arena->buffer))) {
        return false;
    }

    void * mapped = nullptr;
    if (!GGML_HRX_CHECK(hrx_buffer_map(arena->buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, capacity, &mapped))) {
        hrx_buffer_release(arena->buffer);
        arena->buffer = nullptr;
        return false;
    }
    arena->mapped = static_cast<uint8_t *>(mapped);
    arena->capacity = capacity;
    arena->offset = 0;
    return true;
}

static bool ggml_backend_hrx_stage_and_copy_tensor(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t buffer_offset,
        size_t buffer_size,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: upload for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor upload\n", __func__);
        return false;
    }

    std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
    auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
    if (!arena ||
        !ggml_backend_hrx_ensure_staging_buffer_locked(context->device_context, arena, ggml_backend_hrx_staging_arena_capacity())) {
        hrx_stream_release(stream);
        return false;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t uploaded = 0;
    bool ok = true;
    while (uploaded < size) {
        size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
        if (staging_offset >= arena->capacity) {
            ok = GGML_HRX_CHECK(hrx_stream_flush(stream)) && GGML_HRX_CHECK(hrx_stream_wait(stream));
            if (!ok) {
                break;
            }
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
            staging_offset = 0;
        }

        const size_t available = arena->capacity - staging_offset;
        const size_t chunk_size = std::min(size - uploaded, available);
        if (chunk_size == 0) {
            GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
            ok = false;
            break;
        }

        std::memcpy(arena->mapped + staging_offset, bytes + uploaded, chunk_size);
        ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
            stream,
            arena->buffer,
            staging_offset,
            context->buffer,
            buffer_offset + uploaded,
            chunk_size));
        if (!ok) {
            break;
        }

        arena->offset = ggml_backend_hrx_align_up(staging_offset + chunk_size, GGML_HRX_ALIGNMENT);
        uploaded += chunk_size;
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_copy_tensor_to_staging(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * data,
        size_t size,
        const char * reason) {
    GGML_UNUSED(reason);
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: readback for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor readback\n", __func__);
        return false;
    }

    auto * out_bytes = static_cast<uint8_t *>(data);
    size_t copied = 0;
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
        if (!arena ||
            !ggml_backend_hrx_ensure_staging_buffer_locked(context->device_context, arena, ggml_backend_hrx_staging_arena_capacity())) {
            hrx_stream_release(stream);
            return false;
        }

        while (copied < size) {
            size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
            if (staging_offset >= arena->capacity) {
                ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
                if (!ok) {
                    break;
                }
                ggml_backend_hrx_reset_staging_arena_locked(*arena);
                staging_offset = 0;
            }

            const size_t chunk_size = std::min(size - copied, arena->capacity - staging_offset);
            if (chunk_size == 0) {
                GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
                ok = false;
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
                stream,
                context->buffer,
                buffer_offset + copied,
                arena->buffer,
                staging_offset,
                chunk_size));
            if (!ok) {
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
            if (!ok) {
                break;
            }
            std::memcpy(out_bytes + copied, arena->mapped + staging_offset, chunk_size);
            copied += chunk_size;
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static size_t ggml_backend_hrx_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
            &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx_device_architecture(hrx_device_t device) {
    std::array<char, 128> architecture = {};
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        return std::string();
    }
    return std::string(architecture.data());
}

static std::string ggml_backend_hrx_device_description(hrx_device_t device) {
    std::array<char, 128> name = {};
    std::array<char, 128> architecture = {};

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_NAME, name.data(), name.size()))) {
        std::snprintf(name.data(), name.size(), "unknown");
    }

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        std::snprintf(architecture.data(), architecture.size(), "unknown");
    }

    std::string description(name.data());
    if (!description.empty() && architecture[0] != '\0') {
        description += " (";
        description += architecture.data();
        description += ")";
    }
    return description.empty() ? std::string("HRX GPU") : description;
}

static const char * ggml_backend_hrx_kernel_gfx_target(const ggml_backend_hrx_device_context * device_context) {
    GGML_UNUSED(device_context);
    return "gfx1100";
}

static bool ggml_backend_hrx_load_catalog_provider(
        ggml_backend_hrx_device_context * device_context,
        const char * name,
        ggml_backend_hrx_op_provider * provider) {
    const ggml_hrx_kernel_entry * entry =
        ggml_hrx_kernel_catalog_find(name, ggml_backend_hrx_kernel_gfx_target(device_context));
    if (!entry || !entry->data || entry->data_size == 0) {
        return false;
    }

    hrx_executable_t executable = nullptr;
    if (!GGML_HRX_CHECK(hrx_executable_load_data(
            device_context->device, entry->data, entry->data_size, entry->format, &executable))) {
        GGML_LOG_WARN("%s: failed to load HRX catalog kernel %s for %s\n",
            __func__, entry->name, ggml_backend_hrx_kernel_gfx_target(device_context));
        return false;
    }

    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    const bool ok = GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(
                        executable, entry->name, &export_ordinal)) &&
                    GGML_HRX_CHECK(hrx_executable_export_info(
                        executable, export_ordinal, &export_info)) &&
                    export_info.binding_count == entry->binding_count &&
                    export_info.parameter_count == entry->binding_count + 1 &&
                    export_info.constant_count * sizeof(uint32_t) == entry->constants_size &&
                    entry->constants_size == sizeof(ggml_backend_hrx_elementwise_constants);
    if (!ok) {
        GGML_LOG_WARN(
            "%s: HRX catalog kernel %s has unsupported ABI "
            "(bindings=%u expected=%u constants=%u constants_size=%u parameters=%u workgroup=%ux%ux%u)\n",
            __func__,
            entry->name,
            export_info.binding_count,
            entry->binding_count,
            export_info.constant_count,
            entry->constants_size,
            export_info.parameter_count,
            export_info.workgroup_size[0],
            export_info.workgroup_size[1],
            export_info.workgroup_size[2]);
        hrx_executable_release(executable);
        return false;
    }

    provider->kind = ggml_backend_hrx_provider_kind::hsaco;
    provider->executable = executable;
    provider->export_ordinal = export_ordinal;
    provider->export_info = export_info;
    provider->export_info.workgroup_size[0] = entry->workgroup_size[0];
    provider->export_info.workgroup_size[1] = entry->workgroup_size[1];
    provider->export_info.workgroup_size[2] = entry->workgroup_size[2];
    return true;
}

static bool ggml_backend_hrx_load_add_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_add_f32", &device_context->add_provider);
}

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, buffer_offset, size, &value, sizeof(value));
}

static void ggml_backend_hrx_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_stage_and_copy_tensor(context, tensor, data, buffer_offset, buffer->size, size)) {
        GGML_LOG_ERROR("%s: failed to upload tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_copy_tensor_to_staging(
            context, tensor, buffer_offset, buffer->size, data, size, "get_tensor")) {
        GGML_LOG_ERROR("%s: failed to read tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_hrx_get_buffer_context(buffer);
    auto * src_context = ggml_backend_hrx_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    if (!ggml_backend_hrx_sync_streams(dst_context->device_context)) {
        return false;
    }

    const size_t src_offset = ggml_backend_hrx_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_hrx_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    return ggml_backend_hrx_queue_copy_stream_sync(
        dst_context->device_context,
        src_context->buffer, src_offset,
        dst_context->buffer, dst_offset,
        size);
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, 0, buffer->size, &value, sizeof(value));
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);

    hrx_buffer_t hrx_buffer = nullptr;
    if (size > 0 &&
        !GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device),
            buft_context->params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE),
    };
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_hrx_buffer_i, context, size);
    if (!buffer) {
        if (context->buffer) {
            hrx_buffer_release(context->buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);
    return buft_context->device_context->memory_total > 0 ?
        buft_context->device_context->memory_total :
        std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx_device_buffer_type(ggml_backend_dev_t dev) {
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_hrx_buffer_type_context>> contexts;

    for (const auto & buft : buffer_types) {
        auto * context = ggml_backend_hrx_get_buft_context(buft.get());
        if (context->device_context == device_context) {
            return buft.get();
        }
    }

    auto * context = new ggml_backend_hrx_buffer_type_context {
        /* .device_context = */ device_context,
        /* .name           = */ device_context->name,
        /* .params         = */ {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        },
    };

    auto * buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_hrx_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };

    contexts.emplace_back(context);
    buffer_types.emplace_back(buft);
    return buft;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx_unregister_stream(context->device_context, context->stream);
        hrx_stream_release(context->stream);
    }
    delete context;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(context->device_context, context->stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
}

static bool ggml_backend_hrx_supports_add(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->add_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static ggml_status ggml_backend_hrx_dispatch_add(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: ADD tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const auto & provider = context->device_context->add_provider;
    ggml_backend_hrx_elementwise_constants constants = {
        /* .n = */ ggml_nelements(dst),
    };
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ {
            workgroup_size,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        context->device_context->active_stream = context->stream;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_ADD:
                if (!ggml_backend_hrx_supports_add(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ADD shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_add(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }

    ggml_backend_hrx_synchronize(backend);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx_i = {
    /* .get_name           = */ ggml_backend_hrx_get_name,
    /* .free               = */ ggml_backend_hrx_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type = ggml_backend_hrx_device_get_type(dev);
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async = */ true,
        /* .host_buffer = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_context {
        /* .device_context = */ device_context,
        /* .stream         = */ stream,
        /* .name           = */ device_context->name,
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx_guid(),
        /* .iface   = */ ggml_backend_hrx_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }

    ggml_backend_hrx_register_stream(device_context, stream);
    return backend;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_ADD:
            return ggml_backend_hrx_supports_add(ggml_backend_hrx_get_device_context(dev), op);
        default:
            return false;
    }
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_hrx_buffer_type_get_name) {
        return false;
    }
    return buft->device == dev;
}

static const ggml_backend_device_i ggml_backend_hrx_device_i = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx_reg_context * ggml_backend_hrx_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_hrx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_i = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx_reg_get_proc_address,
};

static std::unique_ptr<ggml_backend_hrx_reg_context> ggml_backend_hrx_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx_device_context>();
        device_context->device = device;
        device_context->name = std::string(GGML_HRX_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx_device_description(device);
        device_context->architecture = ggml_backend_hrx_device_architecture(device);
        device_context->memory_total = ggml_backend_hrx_total_memory(device);
        if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &device_context->transfer_stream))) {
            hrx_device_release(device);
            continue;
        }
        ggml_backend_hrx_register_stream(device_context.get(), device_context->transfer_stream);
        (void) ggml_backend_hrx_load_add_provider(device_context.get());

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    return reg ? static_cast<int>(ggml_backend_reg_dev_count(reg)) : 0;
}

void ggml_backend_hrx_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        description[0] = '\0';
        return;
    }

    const char * value = ggml_backend_dev_description(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)));
    std::snprintf(description, description_size, "%s", value ? value : "");
}

void ggml_backend_hrx_get_device_memory(int device, size_t * free, size_t * total) {
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context =
        ggml_backend_hrx_create_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx_reg_i,
        /* .context     = */ context.get(),
    };

    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
