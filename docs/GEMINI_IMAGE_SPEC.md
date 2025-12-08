# Gemini 2.5 Flash Image  
## Aspect ratio, resolution, and token list

| Aspect ratio | Resolution | Tokens |
|--------------|------------|--------|
| 1:1  | 1024 × 1024 | 1290 |
| 2:3  | 832 × 1248  | 1290 |
| 3:2  | 1248 × 832  | 1290 |
| 3:4  | 864 × 1184  | 1290 |
| 4:3  | 1184 × 864  | 1290 |
| 4:5  | 896 × 1152  | 1290 |
| 5:4  | 1152 × 896  | 1290 |
| 9:16 | 768 × 1344  | 1290 |
| 16:9 | 1344 × 768  | 1290 |
| 21:9 | 1536 × 672  | 1290 |

---

# Gemini 3 Pro image preview  
## Aspect ratio, resolution, and token list

### 1K / 2K / 4K mapping

| Aspect ratio | 1K resolution | 1K tokens | 2K resolution | 2K tokens | 4K resolution | 4K tokens |
|--------------|---------------|-----------|---------------|-----------|---------------|-----------|
| 1:1  | 1024 × 1024 | 1210 | 2048 × 2048 | 1210 | 4096 × 4096 | 2000 |
| 2:3  | 848 × 1264  | 1210 | 1696 × 2528 | 1210 | 3392 × 5056 | 2000 |
| 3:2  | 1264 × 848  | 1210 | 2528 × 1696 | 1210 | 5056 × 3392 | 2000 |
| 3:4  | 896 × 1200  | 1210 | 1792 × 2400 | 1210 | 3584 × 4800 | 2000 |
| 4:3  | 1200 × 896  | 1210 | 2400 × 1792 | 1210 | 4800 × 3584 | 2000 |
| 4:5  | 928 × 1152  | 1210 | 1856 × 2304 | 1210 | 3712 × 4608 | 2000 |
| 5:4  | 1152 × 928  | 1210 | 2304 × 1856 | 1210 | 4608 × 3712 | 2000 |
| 9:16 | 768 × 1376  | 1210 | 1536 × 2752 | 1210 | 3072 × 5504 | 2000 |
| 16:9 | 1376 × 768  | 1210 | 2752 × 1536 | 1210 | 5504 × 3072 | 2000 |
| 21:9 | 1584 × 672  | 1210 | 3168 × 1344 | 1210 | 6336 × 2688 | 2000 |

# code samples
```python
message = "Update this infographic to be in Spanish. Do not change any other elements of the image."
aspect_ratio = "16:9" # "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
resolution = "2K" # "1K", "2K", "4K"

response = chat.send_message(message,
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution
        ),
    ))

for part in response.parts:
    if part.text is not None:
        print(part.text)
    elif image:= part.as_image():
        image.save("photosynthesis_spanish.png")
```