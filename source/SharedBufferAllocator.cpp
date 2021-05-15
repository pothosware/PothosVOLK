// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include "SharedBufferAllocator.hpp"

#include <volk/volk.h>
#include <volk/volk_malloc.h>

Pothos::SharedBuffer volkSharedBufferAllocator(const Pothos::BufferManagerArgs& args)
{
    const auto totalSize = args.bufferSize * args.numBuffers;
    auto sharedMem = std::shared_ptr<void>(
        volk_malloc(totalSize, volk_get_alignment()),
        volk_free);

    return Pothos::SharedBuffer(
        reinterpret_cast<size_t>(sharedMem.get()),
        totalSize,
        sharedMem);
}
