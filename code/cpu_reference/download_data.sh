#!/usr/bin/env bash
# Reproduces the exact real-data downloads used for CIFAR-10, Fashion-MNIST,
# and Tiny-ImageNet-200 (15-class subset) in this project's session.
# Run this from the acat_project/ root. Requires curl.
set -e
mkdir -p data_raw

# ---------------- CIFAR-10 (train: 80/class, test: 20/class) ----------------
mkdir -p data_raw/cifar10_raw data_raw/cifar10_raw_test
classes="airplane automobile bird cat deer dog frog horse ship truck"
cd data_raw/cifar10_raw
: > urls.txt
for c in $classes; do
  mkdir -p "$c"
  for i in $(seq 1 80); do
    n=$(printf "%04d" "$i")
    echo "https://raw.githubusercontent.com/YoongiKim/CIFAR-10-images/master/train/$c/$n.jpg $c/$n.jpg" >> urls.txt
  done
done
cat urls.txt | xargs -P 20 -n 2 sh -c 'curl -sS --max-time 20 -o "$1" "$0" || rm -f "$1"'
cd ../cifar10_raw_test
: > urls.txt
for c in $classes; do
  mkdir -p "$c"
  for i in $(seq 1 20); do
    n=$(printf "%04d" "$i")
    echo "https://raw.githubusercontent.com/YoongiKim/CIFAR-10-images/master/test/$c/$n.jpg $c/$n.jpg" >> urls.txt
  done
done
cat urls.txt | xargs -P 20 -n 2 sh -c 'curl -sS --max-time 20 -o "$1" "$0" || rm -f "$1"'
cd ../..

# ---------------- Fashion-MNIST (full official dataset) ----------------
mkdir -p data_raw/fashionmnist_raw
cd data_raw/fashionmnist_raw
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sS -o "$f" --max-time 60 "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f"
done
cd ../..

# ---------------- Tiny-ImageNet-200 (15-class, 50-img/class subset) ----------------
mkdir -p data_raw/tinyimagenet_raw/images
cd data_raw/tinyimagenet_raw
curl -sS --max-time 30 -o wnids.txt "https://raw.githubusercontent.com/seshuad/IMagenet/master/tiny-imagenet-200/wnids.txt"
curl -sS --max-time 30 -o words.txt "https://raw.githubusercontent.com/seshuad/IMagenet/master/tiny-imagenet-200/words.txt"
head -15 wnids.txt > classes15.txt
: > urls.txt
while read -r wnid; do
  mkdir -p "images/$wnid"
  for i in $(seq 0 49); do
    echo "https://raw.githubusercontent.com/seshuad/IMagenet/master/tiny-imagenet-200/train/$wnid/images/${wnid}_${i}.JPEG images/$wnid/${wnid}_${i}.JPEG" >> urls.txt
  done
done < classes15.txt
cat urls.txt | xargs -P 20 -n 2 sh -c 'curl -sS --max-time 20 -o "$1" "$0" || rm -f "$1"'
cd ../..

echo "Done. See src/data.py for how these are loaded."
