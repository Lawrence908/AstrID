# **AstrID**: *ML*
## Full Catalog Training: Best results

```bash
(.venv) chris@daedalus:~/AstrID$ python scripts/train_real_bogus_cnn.py \
  --triplet-dir output/datasets/full_catalog/training_triplets_quality_aug \
  --output-dir output/models/real_bogus_cnn_full_aug \
  --epochs 100 --batch-size 32
INFO:src.core.db.session:No SSL certificate path provided, using certifi CA bundle
INFO:src.core.db.session:Using certifi CA bundle for SSL verification
INFO:src.core.db.session:Creating database engine with URL: postgresql+asyncpg://postgres.vqplumkrlkgrsnnkptqp:****@aws-1-us-west-1.pooler.supabase.com/postgres
INFO:src.core.db.session:Database engine created successfully
INFO:__main__:Loading triplets from output/datasets/full_catalog/training_triplets_quality_aug
INFO:__main__:Samples: 600 real, 528 bogus, total 1128
INFO:__main__:Epoch   1  loss=0.6230  val P=0.852 R=0.552 F1=0.670 AUCPR=0.858
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   2  loss=0.5486  val P=0.641 R=0.856 F1=0.733 AUCPR=0.885
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   3  loss=0.5223  val P=0.891 R=0.656 F1=0.756 AUCPR=0.896
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   4  loss=0.4807  val P=0.940 R=0.632 F1=0.756 AUCPR=0.908
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   5  loss=0.4450  val P=0.924 R=0.680 F1=0.783 AUCPR=0.901
INFO:__main__:Epoch   6  loss=0.4320  val P=0.871 R=0.704 F1=0.779 AUCPR=0.914
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   7  loss=0.4331  val P=0.955 R=0.672 F1=0.789 AUCPR=0.922
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   8  loss=0.3918  val P=0.936 R=0.704 F1=0.804 AUCPR=0.926
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch   9  loss=0.3831  val P=0.899 R=0.712 F1=0.795 AUCPR=0.926
INFO:__main__:Epoch  10  loss=0.3480  val P=0.913 R=0.752 F1=0.825 AUCPR=0.929
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  11  loss=0.3431  val P=0.802 R=0.840 F1=0.820 AUCPR=0.931
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  12  loss=0.3117  val P=0.899 R=0.784 F1=0.838 AUCPR=0.940
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  13  loss=0.2659  val P=0.800 R=0.832 F1=0.816 AUCPR=0.940
INFO:__main__:Epoch  14  loss=0.2545  val P=0.651 R=0.912 F1=0.760 AUCPR=0.925
INFO:__main__:Epoch  15  loss=0.2644  val P=0.870 R=0.800 F1=0.833 AUCPR=0.949
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  16  loss=0.1929  val P=0.923 R=0.768 F1=0.838 AUCPR=0.952
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  17  loss=0.1505  val P=0.844 R=0.824 F1=0.834 AUCPR=0.937
INFO:__main__:Epoch  18  loss=0.1209  val P=0.958 R=0.728 F1=0.827 AUCPR=0.953
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  19  loss=0.0862  val P=0.925 R=0.784 F1=0.848 AUCPR=0.958
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  20  loss=0.0823  val P=0.960 R=0.768 F1=0.853 AUCPR=0.958
INFO:__main__:Epoch  21  loss=0.0612  val P=0.870 R=0.856 F1=0.863 AUCPR=0.953
INFO:__main__:Epoch  22  loss=0.0750  val P=0.873 R=0.824 F1=0.848 AUCPR=0.944
INFO:__main__:Epoch  23  loss=0.0740  val P=0.899 R=0.856 F1=0.877 AUCPR=0.954
INFO:__main__:Epoch  24  loss=0.1216  val P=0.873 R=0.880 F1=0.876 AUCPR=0.955
INFO:__main__:Epoch  25  loss=0.0516  val P=0.932 R=0.768 F1=0.842 AUCPR=0.959
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  26  loss=0.0217  val P=0.944 R=0.808 F1=0.871 AUCPR=0.958
INFO:__main__:Epoch  27  loss=0.0084  val P=0.935 R=0.800 F1=0.862 AUCPR=0.960
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  28  loss=0.0193  val P=0.921 R=0.840 F1=0.879 AUCPR=0.958
INFO:__main__:Epoch  29  loss=0.0151  val P=0.919 R=0.816 F1=0.864 AUCPR=0.957
INFO:__main__:Epoch  30  loss=0.0074  val P=0.920 R=0.824 F1=0.869 AUCPR=0.965
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  31  loss=0.0036  val P=0.922 R=0.848 F1=0.883 AUCPR=0.965
INFO:__main__:Epoch  32  loss=0.0024  val P=0.928 R=0.824 F1=0.873 AUCPR=0.964
INFO:__main__:Epoch  33  loss=0.0016  val P=0.935 R=0.808 F1=0.867 AUCPR=0.964
INFO:__main__:Epoch  34  loss=0.0025  val P=0.928 R=0.824 F1=0.873 AUCPR=0.966
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  35  loss=0.0012  val P=0.906 R=0.848 F1=0.876 AUCPR=0.966
INFO:__main__:Epoch  36  loss=0.0011  val P=0.944 R=0.816 F1=0.876 AUCPR=0.966
INFO:__main__:Epoch  37  loss=0.0011  val P=0.920 R=0.832 F1=0.874 AUCPR=0.965
INFO:__main__:Epoch  38  loss=0.0016  val P=0.936 R=0.816 F1=0.872 AUCPR=0.965
INFO:__main__:Epoch  39  loss=0.0012  val P=0.936 R=0.816 F1=0.872 AUCPR=0.963
INFO:__main__:Epoch  40  loss=0.0018  val P=0.930 R=0.848 F1=0.887 AUCPR=0.966
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  41  loss=0.0022  val P=0.943 R=0.792 F1=0.861 AUCPR=0.963
INFO:__main__:Epoch  42  loss=0.0015  val P=0.944 R=0.808 F1=0.871 AUCPR=0.967
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  43  loss=0.0008  val P=0.920 R=0.832 F1=0.874 AUCPR=0.967
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  44  loss=0.0007  val P=0.920 R=0.832 F1=0.874 AUCPR=0.967
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  45  loss=0.0006  val P=0.936 R=0.824 F1=0.877 AUCPR=0.968
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  46  loss=0.0006  val P=0.929 R=0.832 F1=0.878 AUCPR=0.968
INFO:__main__:  -> saved best checkpoint to output/models/real_bogus_cnn_full_aug/best.pt
INFO:__main__:Epoch  47  loss=0.0005  val P=0.937 R=0.832 F1=0.881 AUCPR=0.968
INFO:__main__:Epoch  48  loss=0.0005  val P=0.929 R=0.832 F1=0.878 AUCPR=0.967
INFO:__main__:Epoch  49  loss=0.0006  val P=0.929 R=0.832 F1=0.878 AUCPR=0.966
INFO:__main__:Epoch  50  loss=0.0006  val P=0.945 R=0.824 F1=0.880 AUCPR=0.968
INFO:__main__:Epoch  51  loss=0.0005  val P=0.937 R=0.832 F1=0.881 AUCPR=0.967
INFO:__main__:Epoch  52  loss=0.0004  val P=0.936 R=0.824 F1=0.877 AUCPR=0.968
INFO:__main__:Epoch  53  loss=0.0004  val P=0.929 R=0.832 F1=0.878 AUCPR=0.967
INFO:__main__:Epoch  54  loss=0.0004  val P=0.929 R=0.832 F1=0.878 AUCPR=0.967
INFO:__main__:Epoch  55  loss=0.0004  val P=0.927 R=0.816 F1=0.868 AUCPR=0.967
INFO:__main__:Epoch  56  loss=0.0004  val P=0.929 R=0.832 F1=0.878 AUCPR=0.967
INFO:__main__:Epoch  57  loss=0.0002  val P=0.928 R=0.824 F1=0.873 AUCPR=0.967
INFO:__main__:Epoch  58  loss=0.0003  val P=0.928 R=0.824 F1=0.873 AUCPR=0.967
INFO:__main__:Epoch  59  loss=0.0004  val P=0.920 R=0.832 F1=0.874 AUCPR=0.968
INFO:__main__:Epoch  60  loss=0.0005  val P=0.920 R=0.832 F1=0.874 AUCPR=0.967
INFO:__main__:Epoch  61  loss=0.0003  val P=0.944 R=0.816 F1=0.876 AUCPR=0.966
INFO:__main__:Early stopping: no val AUCPR improvement for 15 epochs
INFO:__main__:Done. Best val AUCPR: 0.968
INFO:__main__:Output: output/models/real_bogus_cnn_full_au
```