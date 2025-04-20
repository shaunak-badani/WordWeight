import { useRef, useEffect, useState } from "react";
import { Canvas, Rect } from "fabric";
import * as fabric from "fabric";
import { Button } from "@/components/ui/button";


const ImageOverlay = (props: any) => {

    const canvasRef = useRef<HTMLCanvasElement | undefined>(undefined);
    const [canvas, setCanvas] = useState<Canvas | null>(null);
    const [rect, setRect] = useState<Rect | null>(null);

    const { image } = props;

    let imageSrc = `data:image/png;base64,${image}`;
    console.log("ImageSrc : ", imageSrc)

    useEffect(() => {
        if (!canvasRef.current) return;
        const initCanvas = new Canvas(canvasRef.current, {
            width: 512,
            height: 512
        })

        const loadedImage = new Image();
        loadedImage.crossOrigin = "anonymous";
        loadedImage.src = imageSrc;

        loadedImage.onload = () => {
            // Create fabric.js image from the loaded HTML Image
            const fabricImage = new fabric.Image(loadedImage, {
              scaleX: initCanvas.width ? initCanvas.width / loadedImage.width : 1,
              scaleY: initCanvas.height ? initCanvas.height / loadedImage.height : 1
            });
            
            // Set as background image
            initCanvas.backgroundImage = fabricImage;
            initCanvas.renderAll();


            const cRect = new Rect({
                top: 100,
                left: 50,
                width: 100,
                height: 60,
                fill: "rgba(81, 0, 255, 0.38)"
            });
            initCanvas.add(cRect);
            setRect(cRect);
            initCanvas.renderAll();
        };


        return () => {
            initCanvas.dispose();
        };
    }, [imageSrc]);

    const handleExplainBox = () => {
        if(canvas)
        {

            const maskCanvas = new fabric.StaticCanvas(null, {
                width: canvas.width,
                height: canvas.height,
                backgroundColor: 'black',
            });
            // console.log(maskCanvas);
            // const cloneRect = fabric.util.object.clone(rect) as Rect;
            const cloneRect = rect;
            if(cloneRect)
            {
                cloneRect.set({
                    fill: 'white',
                    stroke: null,
                    opacity: 1,
                });
                console.log(cloneRect);
                maskCanvas.add(cloneRect);
            }
            maskCanvas.renderAll();

            const p = maskCanvas.toDataURL({ format: 'png' });
            console.log("p : ", p);
        }
    }

    return (
        <>
            <div className="flex justify-center">
                <canvas ref={canvasRef} width={500} height={500} style={{ border: "1px solid #ccc" }} />
                
            </div>
            <Button className="p-6 sm:p-6 rounded-2xl m-8 sm:m-8" onClick={handleExplainBox}>
                Explain
            </Button>
        </>
    )

}

export default ImageOverlay;