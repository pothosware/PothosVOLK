// Copyright 2021 Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

#include <Pothos/Plugin.hpp>

#include <Poco/Logger.h>

#include <volk/volk_prefs.h>

pothos_static_block(pothosVOLKCheckVOLKConfig)
{
    constexpr size_t VOLKPathSize = 512;
    char path[VOLKPathSize] = {0};

    volk_get_config_path(path, true);
    if(path[0] == 0)
    {
        Poco::Logger::get("PothosVOLK").warning("No VOLK config file found. Run volk_profile for best performance.");
    }
}
