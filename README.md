# Handwritten Chinese Writer Identification

### Open source packages used
- Tkinter for user interface
- Keras for building the network
- Tensorflow for building dataset
- scikit-image for image preprocessing and segmentation
- tesseract for OCR (demonstration only)

I only used the library functions provided in packages. The entire logic of the program (including model design, user interface design) is completely created by myself.

### Requirements
(I'm not sure about specific requirements on the versions of the packages, so I'll just list the environment on my computer.)
```python
Keras==2.4.3
lime==0.2.0.1
numpy==1.19.0
opencv-python==4.4.0.44
Pillow==8.4.0
pytesseract==0.3.8      # tesseract v5.0.0-rc1.20211030 with chi_sim language package
scikit-image==0.18.3
tensorflow==2.2.0       # with GPU support
tkinter==8.6
```

### Acknowledgements
Thank professors Haiyi Zhu and Steven Wu for wonderful lectures that provided inspiration in improving this project. Thank my roommate for providing ideas in character segmentation.