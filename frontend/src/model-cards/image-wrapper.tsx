import { useRef, useEffect, useState } from "react";
import { Canvas, Rect } from "fabric";
import * as fabric from "fabric";
import { Button } from "@/components/ui/button";
import backendClient from "@/backendClient";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import { TokenImportance } from "@/TokenImportance";


const ImageOverlay = (props: any) => {

    const [isLoading, setLoading] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [canvas, setCanvas] = useState<Canvas | null>(null);
    const [rect, setRect] = useState<Rect | null>(null);
    const [maskedImage, setMaskedImage] = useState("");
    const [tokenImportances, setTokenImportances] = useState<TokenImportance[]>([]);

    let fontWeights = [
        "font-thin",
        "font-extralight",
        "font-light",
        "font-normal",
        "font-medium",
        "font-semibold",
        "font-bold",
        "font-extrabold",
        "font-black"
    ]
    // const [responses, ]

    let { image, prompt } = props;
    if(maskedImage)
        image = maskedImage

    let imageSrc = `data:image/png;base64,${image}`;

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
            setCanvas(initCanvas);
            initCanvas.renderAll();
        };


        return () => {
            initCanvas.dispose();
        };
    }, [imageSrc]);

    const handleExplainBox = async() => {
        setLoading(true);
        if(canvas)
        {

            const maskCanvas = new fabric.StaticCanvas(undefined, {
                width: canvas.width,
                height: canvas.height,
                backgroundColor: 'black',
            });
            const cloneRect = rect;
            if(cloneRect)
            {
                cloneRect.set({
                    fill: 'white',
                    stroke: null,
                    opacity: 1,
                });
                maskCanvas.add(cloneRect);
            }
            maskCanvas.renderAll();

            const p = maskCanvas.toDataURL({ format: 'png', 
                quality: 1,
                multiplier: 1,
                left: 0,
                top: 0,
                width: maskCanvas.getWidth(),
                height: maskCanvas.getHeight()
            });

            const response = await backendClient.post("/explain", {
                "prompt": prompt,
                "image_base64": p,
            });

            if(!response.data)
            {
                setLoading(false);
                return;
            }
            setMaskedImage(response.data.masked_image);
            setTokenImportances(response.data.tokens_imp);
            setLoading(false);
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
            {}
            {isLoading && <BackdropWithSpinner />}
            <div className="text-3xl">
                {tokenImportances.map(tokenImp => {
                    const index = Math.floor(tokenImp.importance / 0.11);
                    return(<span className={fontWeights[index]}>{tokenImp.word} </span>);
                })}
            </div>
        </>
    )

}

export default ImageOverlay;