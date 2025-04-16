## GFEraser
Official source code for TOIS 2024 paper: [Efficient and Adaptive Recommendation Unlearning: A Guided Filtering Framework to Erase Outdated Preferences](https://dl.acm.org/doi/10.1145/3706633)


## Datasets

You can use the code in `data process` folder to process your own dataset. We explain its role at beginning of each code file.


## Run the code
You can run our method in the folder corresponding to the base model:

```
src_MF:
python main.py --data_name=Yelp --original_model=MF --cl_weight=0.4 --pos_bpr_weight=0.001 --model_idx=1

src_NGCF:
python main.py --data_name=Yelp --original_model=NGCF --cl_weight=1.0 --pos_bpr_weight=0.0175 --model_idx=1

src_LightGCN:
python main.py --data_name=Yelp --original_model=LightGCN --cl_weight=0.4 --pos_bpr_weight=0.02 --model_idx=1
```


## Acknowledgement

Some models are implemented based on [SELFRec](https://github.com/Coder-Yu/SELFRec).

Thanks for providing efficient implementation.


## Reference

Please cite our paper if you use this code.
```
@article{dang2025efficient,
  title={Efficient and Adaptive Recommendation Unlearning: A Guided Filtering Framework to Erase Outdated Preferences},
  author={Dang, Yizhou and Liu, Yuting and Yang, Enneng and Guo, Guibing and Jiang, Linying and Zhao, Jianzhe and Wang, Xingwei},
  journal={ACM Transactions on Information Systems},
  volume={43},
  number={2},
  pages={1--25},
  year={2025},
  publisher={ACM New York, NY}
}
```
