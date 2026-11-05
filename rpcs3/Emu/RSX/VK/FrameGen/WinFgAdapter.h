// SPDX-License-Identifier: GPL-2.0-only
//
// Adapter that lets VKFrameGen.cpp drive the win-fg engine (MIT, (c) 2026 The412Banner)
// through the SAME four calls it used for the Eden/LSFG port: Process, WantedGenerations,
// GeneratedFrameCount and GenerateInto. Nothing here needs Lossless.dll, vulkanMemoryModel or
// nullDescriptor -- win-fg's shaders are plain Vulkan 1.1 compute and are embedded.
//
// Differences from the LSFG FrameGen worth knowing:
//  * win-fg reads two sampled copies of the real frames (prev, curr). Process() keeps them in a
//    two-image ring; the copy is recorded into the caller's command buffer.
//  * It always generates ONE frame per real frame (x2). Higher multipliers are clamped to x2.
//  * win-fg shares descriptor sets and a UBO between frames, so the caller must not have a
//    previous frame-generation submission still running when record() is called.

#pragma once

#include <array>

#include "FrameGenConfig.h"
#include "FrameGenPacer.h"
#include "FrameGenTypes.h"
#include "LsfgVkCompat.h"
#include "WinFgEngine.hpp"

namespace vk { class render_device; }

namespace Vulkan {

class WinFgAdapter {
public:
    explicit WinFgAdapter(MemoryAllocator& memory_allocator, const ::vk::render_device* device);
    ~WinFgAdapter();

    WinFgAdapter(const WinFgAdapter&) = delete;
    WinFgAdapter& operator=(const WinFgAdapter&) = delete;

    /// Ring slot of the submission being recorded. The caller must have fence-waited it: win-fg
    /// reads back that slot's previous global-motion buffer on the host.
    void SetSlot(u32 slot) {
        slot_ = slot % static_cast<u32>(winfg::FrameGen::kGmSlots);
    }

    void Process(const Device& device, vk::CommandBuffer cmdbuf, VkImage image,
                 VkImageView storage_view, VkExtent2D extent, VkFormat format,
                 VkExtent2D guest_extent);

    [[nodiscard]] size_t WantedGenerations(size_t capacity);
    [[nodiscard]] size_t GeneratedFrameCount() const;

    void GenerateInto(const Device& device, vk::CommandBuffer cmdbuf, VkImage image,
                      VkImageView storage_view, size_t generation);

private:
    struct Frame {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
    };

    bool InitEngine();
    bool Rebuild(VkExtent2D extent, VkFormat format);
    bool MakeFrame(Frame& frame, VkExtent2D extent, VkFormat format);
    void DestroyFrames();
    void CopyIn(VkCommandBuffer cmd, VkImage source, Frame& destination, VkExtent2D extent);
    void WaitForIdle();

    const ::vk::render_device* gs_device;

    winfg::DeviceDispatch dd{};
    winfg::InstanceDispatch id{};
    winfg::FrameGen engine;

    FrameGenPacer pacer;
    FrameGenPlan plan{};

    std::array<Frame, 2> frames{};
    VkExtent2D built_extent{};
    VkFormat built_format{VK_FORMAT_UNDEFINED};

    u64 count{};
    u32 slot_{};
    bool engine_ready{};
    bool frames_ready{};
    bool generated{};
    bool unavailable{};
};

} // namespace Vulkan
