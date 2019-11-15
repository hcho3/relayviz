import tvm
import tvm.relay as relay
import tvm.relay.testing as testing

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)

mod, params = testing.resnet.get_workload(
    num_layers=18, batch_size=batch_size, image_shape=image_shape)

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    if isinstance(node, relay.op.op.Op):
        return 
    node_dict[node] = len(node_dict)

node_dict = {}
relay.analysis.post_order_visit(mod['main'], lambda node: _traverse_expr(node, node_dict))
for node, node_idx in node_dict.items():
    if isinstance(node, relay.expr.Var):
        print(f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
    elif isinstance(node, relay.expr.Call):
        args = [node_dict[arg] for arg in node.args]
        print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
    elif isinstance(node, relay.expr.Function):
        print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
    elif isinstance(node, relay.expr.TupleGetItem):
        print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
    else:
        print(f'node_idx: {node_idx}, node: {type(node)}')

