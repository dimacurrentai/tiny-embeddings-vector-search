"""Write the cached ONNX model's tensors and vocabulary in a flat layout for embed.c.

The ONNX file is a protobuf message. Only three nested messages matter here
(ModelProto.graph, GraphProto.initializer, and TensorProto fields), so this
script decodes the wire format directly instead of depending on the onnx package.

Output, next to the cached model under c/:
  weights.bin  magic "MLM6", uint32 tensor count, then for each tensor a
               uint32 name length, the UTF-8 name, uint32 rank, uint32 dims,
               and float32 values in row-major order.
  vocab.txt    one WordPiece token per line, in ID order.
"""

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_vector_search.embedding import REVISION  # noqa: E402
from mini_vector_search.server import DEFAULT_DATA  # noqa: E402

HIDDEN, INTERMEDIATE, LAYERS = 384, 1536, 6


def fields(buffer):
    """Yield (field number, wire type, value) for one protobuf message."""
    position = 0
    while position < len(buffer):
        tag, position = varint(buffer, position)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, position = varint(buffer, position)
        elif wire == 2:
            length, position = varint(buffer, position)
            value = buffer[position : position + length]
            position += length
        elif wire == 1:
            value, position = buffer[position : position + 8], position + 8
        elif wire == 5:
            value, position = buffer[position : position + 4], position + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield number, wire, value


def varint(buffer, position):
    result = shift = 0
    while True:
        byte = buffer[position]
        position += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, position
        shift += 7


def tensors(model_bytes):
    """Return {name: (dims, float32 bytes)} for every initializer in the graph."""
    graph = next(value for number, _, value in fields(model_bytes) if number == 7)
    result = {}
    for number, _, initializer in fields(graph):
        if number != 5:
            continue
        dims, data_type, name, raw = [], None, None, None
        for field, wire, value in fields(initializer):
            if field == 1:
                dims.append(value)
            elif field == 2:
                data_type = value
            elif field == 8:
                name = value.decode()
            elif field == 9:
                raw = value
        if data_type != 1 or raw is None:
            raise ValueError(f"{name}: expected float32 raw_data")
        result[name] = (dims, raw)
    return result


def layer_names(index, matmul_ids):
    prefix = f"encoder.layer.{index}."
    roles = ["attention.self.query", "attention.self.key", "attention.self.value",
             "attention.output.dense", "intermediate.dense", "output.dense"]
    ordered = []
    for role, matmul in zip(roles, matmul_ids):
        ordered.append((prefix + role + ".weight", matmul))
        ordered.append((prefix + role + ".bias", prefix + role + ".bias"))
    for norm in ["attention.output.LayerNorm", "output.LayerNorm"]:
        for part in ["weight", "bias"]:
            ordered.append((prefix + norm + "." + part, prefix + norm + "." + part))
    return ordered


def main():
    model_dir = DEFAULT_DATA / "models" / REVISION
    output = model_dir / "c"
    output.mkdir(exist_ok=True)
    found = tensors((model_dir / "onnx/model.onnx").read_bytes())
    # The exporter named projection weights onnx::MatMul_<n>; they appear in
    # graph order, six per layer, so sort by number and assign by position.
    matmuls = sorted((name for name in found if name.startswith("onnx::MatMul_")),
                     key=lambda name: int(name.rsplit("_", 1)[1]))
    assert len(matmuls) == 6 * LAYERS, len(matmuls)
    ordered = [(name, name) for name in [
        "embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias"]]
    for layer in range(LAYERS):
        ordered += layer_names(layer, matmuls[6 * layer : 6 * layer + 6])
    with (output / "weights.bin").open("wb") as out:
        out.write(b"MLM6" + struct.pack("<I", len(ordered)))
        for public, source in ordered:
            dims, raw = found[source]
            encoded = public.encode()
            out.write(struct.pack("<I", len(encoded)) + encoded)
            out.write(struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}I", *dims))
            out.write(raw)
    vocab = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    tokens = sorted(vocab, key=vocab.get)
    assert [vocab[token] for token in tokens] == list(range(len(tokens)))
    (output / "vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    print(f"Wrote {len(ordered)} tensors and {len(tokens)} tokens to {output}")


if __name__ == "__main__":
    main()
