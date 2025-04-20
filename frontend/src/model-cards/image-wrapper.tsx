import { useRef, useEffect, useState } from "react";
import { Canvas, Rect } from "fabric";
import * as fabric from "fabric";


const ImageOverlay = (props: any) => {

    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fabricCanvasRef = useRef<Canvas | null>(null);
    const [canvas, setCanvas] = useState<Canvas | null>(null);

    const { image } = props;

    let imageSrc = `data:image/png;base64,${image}`;

    const currentImage = new Image();
    currentImage.src = imageSrc;
    useEffect(() => {
        if(canvasRef.current)
        {
            console.log("Here")
            const initCanvas = new Canvas(canvasRef.current, {
                width: 512,
                height: 512
            })
            // initCanvas.backgroundColor = "#555";
            initCanvas.backgroundImage = new fabric.Image(currentImage);
            // initCanvas.backgroundImage = imageSrc
            setCanvas(initCanvas);

            const rect = new Rect({
                top: 100,
                left: 50,
                width: 100,
                height: 60,
                fill: "#D84D42"
            });
            initCanvas.add(rect);
            initCanvas.renderAll();

            return () => {
                initCanvas.dispose();
            };
        }
    }, []);
    


    return (
        <div className="flex justify-center">
            {/* <img src={imageSrc} /> */}
            <canvas ref={canvasRef} width={500} height={500} style={{ border: "1px solid #ccc" }} />
        </div>
    )

}


export default ImageOverlay;