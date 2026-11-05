// SPDX-License-Identifier: GPL-2.0-only
// See WinFgAdapter.h. The engine it drives is win-fg, MIT licensed (WinFg-LICENSE-MIT.txt).

#include "WinFgAdapter.h"

#include "../vkutils/device.h"
#include "WinFgLog.hpp"

#include <algorithm>

namespace Vulkan {

namespace {

// Every device-level entry point the win-fg engine calls. All of them are in ARMSX3's own
// loader (vk_android_loader.h); the swapchain / present ones win-fg's *layer* uses are not
// needed here and stay null.
void FillDispatch(winfg::DeviceDispatch& d, winfg::InstanceDispatch& i) {
#define WFG_D(name) d.name = vk##name;
    WFG_D(AllocateDescriptorSets)
    WFG_D(AllocateMemory)
    WFG_D(BindBufferMemory)
    WFG_D(BindImageMemory)
    WFG_D(CmdBindDescriptorSets)
    WFG_D(CmdBindPipeline)
    WFG_D(CmdClearColorImage)
    WFG_D(CmdCopyImage)
    WFG_D(CmdDispatch)
    WFG_D(CmdPipelineBarrier)
    WFG_D(CreateBuffer)
    WFG_D(CreateComputePipelines)
    WFG_D(CreateDescriptorPool)
    WFG_D(CreateDescriptorSetLayout)
    WFG_D(CreateImage)
    WFG_D(CreateImageView)
    WFG_D(CreatePipelineLayout)
    WFG_D(CreateSampler)
    WFG_D(CreateShaderModule)
    WFG_D(DestroyBuffer)
    WFG_D(DestroyDescriptorPool)
    WFG_D(DestroyDescriptorSetLayout)
    WFG_D(DestroyImage)
    WFG_D(DestroyImageView)
    WFG_D(DestroyPipeline)
    WFG_D(DestroyPipelineLayout)
    WFG_D(DestroySampler)
    WFG_D(DestroyShaderModule)
    WFG_D(DeviceWaitIdle)
    WFG_D(FreeMemory)
    WFG_D(GetBufferMemoryRequirements)
    WFG_D(GetImageMemoryRequirements)
    WFG_D(MapMemory)
    WFG_D(ResetDescriptorPool)
    WFG_D(UnmapMemory)
    WFG_D(UpdateDescriptorSets)
#undef WFG_D
    i.GetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
}

} // namespace

WinFgAdapter::WinFgAdapter(MemoryAllocator&, const ::vk::render_device* device)
    : gs_device{device} {}

WinFgAdapter::~WinFgAdapter() {
    if (engine_ready || frames_ready) {
        WaitForIdle();
        DestroyFrames();
        if (engine_ready) {
            engine.destroy();
        }
    }
}

void WinFgAdapter::WaitForIdle() {
    if (dd.DeviceWaitIdle) {
        dd.DeviceWaitIdle(static_cast<VkDevice>(*gs_device));
    }
}

bool WinFgAdapter::InitEngine() {
    FillDispatch(dd, id);

    const VkDevice device = *gs_device;
    const VkPhysicalDevice phys = gs_device->gpu();

    if (!engine.init(&dd, &id, phys, device, gs_device->get_graphics_queue_family(),
                     gs_device->get_graphics_queue())) {
        WFG_LOGE("WinFgAdapter: engine init failed");
        return false;
    }

    winfg::Config cfg;
    cfg.enabled = true;
    cfg.model = 3;          // symmetric optical flow: the cheaper model, right for GPU-bound Mali
    cfg.multiplier = 2;
    cfg.perfPreset = ::vk::lsfg::performance_mode() ? 2 : 1;   // 2 = coarsest, cheapest search
    engine.configure(cfg);

    engine_ready = true;
    WFG_LOGI("WinFgAdapter: engine ready (model=%d, perf_preset=%d)", cfg.model, cfg.perfPreset);
    return true;
}

bool WinFgAdapter::MakeFrame(Frame& frame, VkExtent2D extent, VkFormat format) {
    const VkDevice device = *gs_device;

    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = {extent.width, extent.height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(device, &ci, nullptr, &frame.image) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device, frame.image, &req);

    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(gs_device->gpu(), &props);

    u32 type = ~0u;
    for (u32 t = 0; t < props.memoryTypeCount; ++t) {
        if ((req.memoryTypeBits & (1u << t)) &&
            (props.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            type = t;
            break;
        }
    }
    if (type == ~0u) {
        return false;
    }

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &ai, nullptr, &frame.memory) != VK_SUCCESS ||
        vkBindImageMemory(device, frame.image, frame.memory, 0) != VK_SUCCESS) {
        return false;
    }

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = frame.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return vkCreateImageView(device, &vi, nullptr, &frame.view) == VK_SUCCESS;
}

void WinFgAdapter::DestroyFrames() {
    const VkDevice device = *gs_device;
    for (Frame& f : frames) {
        if (f.view) vkDestroyImageView(device, f.view, nullptr);
        if (f.image) vkDestroyImage(device, f.image, nullptr);
        if (f.memory) vkFreeMemory(device, f.memory, nullptr);
        f = Frame{};
    }
    frames_ready = false;
}

bool WinFgAdapter::Rebuild(VkExtent2D extent, VkFormat format) {
    // The old frames and the engine's per-size images may still be in flight.
    WaitForIdle();
    DestroyFrames();

    if (!engine.onResize(extent, format)) {
        WFG_LOGE("WinFgAdapter: engine resize to %ux%u failed", extent.width, extent.height);
        return false;
    }
    for (Frame& f : frames) {
        if (!MakeFrame(f, extent, format)) {
            DestroyFrames();
            return false;
        }
    }

    frames_ready = true;
    built_extent = extent;
    built_format = format;
    count = 0;
    pacer.Reset();
    WFG_LOGI("WinFgAdapter: built for %ux%u", extent.width, extent.height);
    return true;
}

void WinFgAdapter::CopyIn(VkCommandBuffer cmd, VkImage source, Frame& dst, VkExtent2D extent) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dst.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    // The whole image is overwritten, so the old contents can be discarded (UNDEFINED).
    // srcStage COMPUTE: the previous frame's flow/synth passes may still be reading it.
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);

    VkImageCopy region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.extent = {extent.width, extent.height, 1};
    // The caller has already put the presented image in GENERAL and made its writes visible.
    vkCmdCopyImage(cmd, source, VK_IMAGE_LAYOUT_GENERAL, dst.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // win-fg declares its prev/curr descriptors as SHADER_READ_ONLY_OPTIMAL.
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
}

void WinFgAdapter::Process(const Device&, vk::CommandBuffer cmdbuf, VkImage image,
                           VkImageView storage_view, VkExtent2D extent, VkFormat format,
                           VkExtent2D) {
    generated = false;

    if (unavailable || !::vk::lsfg::enabled()) {
        if (frames_ready) {
            WaitForIdle();
            DestroyFrames();
            count = 0;
        }
        return;
    }

    if (storage_view == VK_NULL_HANDLE) {
        unavailable = true;
        return;
    }

    if (!engine_ready && !InitEngine()) {
        unavailable = true;
        return;
    }

    if (!frames_ready || built_extent.width != extent.width ||
        built_extent.height != extent.height || built_format != format) {
        if (!Rebuild(extent, format)) {
            unavailable = true;
            return;
        }
    }

    CopyIn(*cmdbuf, image, frames[count & 1u], extent);
    ++count;
    generated = count >= 2;
}

size_t WinFgAdapter::WantedGenerations(size_t capacity) {
    if (unavailable) {
        plan = {};
        return 0;
    }
    // win-fg makes one frame per real frame; telling the pacer the capacity is 1 keeps its
    // own bookkeeping consistent with what is really issued.
    plan = pacer.Plan(std::min<size_t>(capacity, 1));
    return plan.generations;
}

size_t WinFgAdapter::GeneratedFrameCount() const {
    return (generated && plan.warm) ? std::min<size_t>(plan.generations, 1) : 0;
}

void WinFgAdapter::GenerateInto(const Device&, vk::CommandBuffer cmdbuf, VkImage,
                                VkImageView storage_view, size_t generation) {
    if (!frames_ready || !engine.valid() || count < 2) {
        return;
    }

    const u32 curr = static_cast<u32>((count - 1) & 1u);
    const u32 prev = curr ^ 1u;
    const f32 alpha = static_cast<f32>(generation + 1) /
                      static_cast<f32>(std::max<size_t>(plan.generations, 1) + 1);

    engine.record(*cmdbuf, frames[prev].view, frames[curr].view, storage_view, alpha, slot_);
}

} // namespace Vulkan
