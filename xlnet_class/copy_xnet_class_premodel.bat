@echo off
rem echo aaa
set res_path=E:\pytorch_transformer\24197ba0ce5dbfe23924431610704c88e2c0371afa49149360e4c823219ab474.7eac4fe898a021204e63c88c00ea68c60443c57f94b4bc3c02adbde6465745ac
set tar_path=C:\Users\cyd\.cache\torch\pytorch_transformers
echo %res_path%
echo %tar_path%
copy %res_path% %tar_path% /z /y
