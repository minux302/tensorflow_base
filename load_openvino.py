
from openvino.inference_engine import IENetwork, IEPlugin

chip = "CPU"
ir_path = "hoge"
plugin = IEPlugin(device=chip, plugin_dirs=None)

if chip == 'CPU':
  model_xml = ir_path + str(".xml")
  model_bin = ir_path + str(".bin")
else:
  model_xml = ir_path + "_myriad" + str(".xml")
  model_bin = ir_path + "_myriad" + str(".bin")

net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = plugin.load(network=net)
