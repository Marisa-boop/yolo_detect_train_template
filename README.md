1. add data to datasets_origin
```
datasets_origin
- 1
  - images
  - masks
- 2
  - images
  - masks
...
```

2. `python ./utils/merge_data.py`
3. `python ./utils/masks2yoloForm.py`
4. `python ./utils/spilit_data.py`
4. `python ./utils/train.py`
