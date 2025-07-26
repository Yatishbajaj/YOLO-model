from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('runs/detect/train6/weights/last.pt')

    results = model.train(
        data='data/KIIT-MiTA/KIIT-MiTA.yaml',
        epochs=50,
        imgsz=640,  
        batch=16,
        resume=True 
    )

    print("\nTraining complete!")
    print(f"Model weights (best.pt) saved to: {results.save_dir}/weights/best.pt")