import maya.cmds as cmds
import os


# Path to your plugin (.mll)
plugin_path = r"C:\Users\Geri\Documents\Projects\CG\MayaGaussianSplatting\out\build\x64-Release\MayaGaussianSplatting.mll"


# Load plugin
plugin_name = os.path.basename(plugin_path)

if not cmds.pluginInfo(plugin_name, query=True, loaded=True):
    print("Loading plugin...")
    cmds.loadPlugin(plugin_path)
else:
    print("Plugin already loaded.")


# Create GaussianSplattingLocator
locator = cmds.createNode("GaussianSplattingLocator")
print("Created locator:", locator)


print("Done.")