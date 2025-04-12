#!/usr/bin/env/bash

curl "https://www.arcgis.com/sharing/rest/content/items/6cb9092a37da4b5ea1b5f8b054c343aa/data" \
    --output IPN_GB_2023.zip

unzip "./IPN_GB_2023.zip"