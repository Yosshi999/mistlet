# Mistlet

Non official onnx-based implementation for [Mist V1](https://github.com/psyker-team/mist) ([paper](https://arxiv.org/abs/2305.12683))

## Implementation detail
- [x] Optimize textual loss against `MIST.png`
- [ ] Optimize semantic loss

## Usage
### Python impl
Type `--help` for detailed options.

```
python reference.py
```

### Generate ONNX files
Some *.onnx are generated under `./app`.

```
python deploy/convert.py
```

### ONNX impl
Type `--help` for detailed options.

```
python main.py
```
