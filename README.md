# Katharsis

This repository provide not only various face-related tools but also FER system.



Installation
---

### Overall
```bash
pip install torch torchvision torchaudio
pip install tensorflow
conda install -c menpo opencv
pip install --user grad-cam==1.4.6
```

### (Dense) Facial Landmarks
```bash
pip install mediapipe==0.10.0
pip install protobuf==4.21
```
[Here](https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png), you can view the index information of a total of 468 facial landmarks.
### Facial Emotion Recognition
```bash
pip install facetorch
pip install timm==0.6.7
```

```bash
git clone https://github.com/HSE-asavchenko/face-emotion-recognition
```


References
---
- [mediapipe](https://github.com/googlesamples/mediapipe)
- [facetorch](https://github.com/tomas-gajarsky/facetorch)
- [face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
