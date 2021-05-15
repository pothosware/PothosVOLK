#!/usr/bin/env python

from mako.template import Template
import mako.exceptions

import datetime
import json
import os
import sys
import yaml

ScriptDir = os.path.dirname(__file__)
OutputDir = os.path.abspath(sys.argv[1])
Now = datetime.datetime.now()

FactoryTemplate = None
BlockExecutionTestAutoTemplate = None

prefix = """// Copyright (c) 2021-{0} Nicholas Corgan
// SPDX-License-Identifier: GPL-3.0-or-later

//
// This file was auto-generated on {1}.
//
""".format(Now.year, Now)

def processYAMLFile(yamlPath):
    yml = None
    with open(yamlPath) as f:
        yml = yaml.load(f.read())

    if not yml:
        raise RuntimeError("No YAML found in {0}".format(yamlPath))

    global FactoryTemplate
    global BlockExecutionTestAutoTemplate

    factoryFunctionTemplatePath = os.path.join(ScriptDir, "Factory.mako.cpp")
    with open(factoryFunctionTemplatePath) as f:
        FactoryTemplate = f.read()

    try:
        rendered = Template(FactoryTemplate).render(
                       oneToOneBlocks=yml["OneToOneBlocks"],
                       oneToOneWithScalarParamBlocks=yml["OneToOneWithScalarParamBlocks"],
                       docs=[])
    except:
        print(mako.exceptions.text_error_template().render())

    print(rendered)

    return yml

if __name__ == "__main__":
    yamlFilepath = os.path.join(ScriptDir, "Blocks.yaml")
    allBlockYAML = processYAMLFile(yamlFilepath)
