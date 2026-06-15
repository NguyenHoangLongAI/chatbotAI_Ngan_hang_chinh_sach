# patch_model.py
import glob
import os

files = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/PaddlePaddle/*/*/modeling_paddleocr_vl.py"
))

if not files:
    print("File not found!")
    exit(1)

f = files[0]
print(f"Patching: {f}")

with open(f, 'r') as fp:
    content = fp.read()

# Backup
with open(f + '.bak', 'w') as fp:
    fp.write(content)

# Patch import
old = "from transformers.masking_utils import create_causal_mask"
new = """try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    import torch
    def create_causal_mask(sequence_length, past_seqlen=0, inputs_embeds=None, **kwargs):
        total = sequence_length + past_seqlen
        mask = torch.tril(torch.ones((sequence_length, total), dtype=torch.bool))
        return mask"""

if old in content:
    content = content.replace(old, new)
    with open(f, 'w') as fp:
        fp.write(content)
    print("✅ Patched successfully")
else:
    print("⚠️  Pattern not found, checking content:")
    for i, line in enumerate(content.split('\n')[25:45], 26):
        print(f"  {i}: {line}")