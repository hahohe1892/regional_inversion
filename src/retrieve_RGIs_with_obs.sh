#!/bin/bash

for glacier in RGI60-08.00434 RGI60-08.01657 RGI60-08.01779 RGI60-08.02666 RGI60-08.01258 RGI60-08.02382 RGI60-08.00966 RGI60-08.00987 RGI60-08.00312 RGI60-08.02972 RGI60-08.01103 RGI60-08.02967 RGI60-08.00213
do
scp alvis2:/mimer/NOBACKUP/groups/snic2022-22-55/regional_inversion/output/$glacier/ex.nc /home/thomas/regional_inversion/output/$glacier/
done
