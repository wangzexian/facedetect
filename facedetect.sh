#!/bin/bash

if [ ! -d "$1" ] || [ ! -d "$2" ] ; then
  echo "Usage: facedetect.sh <input directory> <output directory> <facedetect arguments>";
  exit 1
fi

for file in $1/*.[jJ][pP][gG]; do
  name=$(basename "$file")
  python facedetect.py "$file" -o "$2/$name" ${*:3}
done
