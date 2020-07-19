# Face Detector

A python script that uses Python's OpenCV2 and face-recognition libraries to detect faces in live video or an image 

## Usage

If you want to use the script in live video, make sure to specifiy that in the arguments. Other than that, all you need are the image path container the faces you want learned, the names of the faces, and if you don't want live video, the path to the test image to run the detector on. All of these are passed as arguments.

For example, you can clone this repo and run

```python

python3 face.py --video True --image_dir /path/to/images 

```

Or without video (the default is set to no video)

```python
python3 face.py --image_dir /path/to/images unknown_image_path /path/to/test/image
```
