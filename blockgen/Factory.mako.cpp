#include "VOLKBlock.hpp"

#include <Pothos/Callable.hpp>
#include <Pothos/Framework.hpp>
#include <Pothos/Plugin.hpp>
#include <Pothos/Proxy.hpp>

#include <volk/volk.h>

#include <vector>

static const std::vector<Pothos::BlockRegistry> BlockRegistries =
{
%for block in oneToOneBlocks:
    Pothos::BlockRegistry(
        "/volk/${block["blockName"]}",
        Pothos::Callable(&VOLKBlock::make).bind<VOLKFcn<${block["inType"]}, ${block["outType"]}>>(${block["fcn"]})),
%endfor
%for block in oneToOneWithScalarParamBlocks:
    Pothos::BlockRegistry(
        "/volk/${block["blockName"]}",
        Pothos::Callable(&VOLKScalarParamBlock::make).bind<VOLKFcn<${block["inType"]}, ${block["outType"]}, ${block["scalarType"]}>>(${block["fcn"]})),
%endfor
};

pothos_static_block(registerPothosVOLKDocs)
{
%for doc in docs:
    ${doc},
%endfor
}
