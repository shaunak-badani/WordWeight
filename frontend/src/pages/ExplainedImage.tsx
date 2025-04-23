import { Card, CardContent } from "@/components/ui/card";
import { TokenImportance } from "@/TokenImportance";

const ExplainedImage = (props: any) => {
    const {image, key} = props;

    const imageSrc = image.masked_image;
    const tokenImportances: TokenImportance[] = image.tokens_imp;
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
    return (
        
        <div className="flex justify-center m-6 sm:m-6" key={key}>
        <Card className="w-full flex">
          <img 
            src={`data:image/jpeg;base64,${imageSrc}`} 
            alt="Business"
            style={{ maxWidth: "40%", height: "auto" }} 
            />
          <div className="w-full flex flex-row items-center justify-center text-2xl">
            <CardContent>
                {tokenImportances.map(tokenImp => {
                    const index = Math.floor(tokenImp.importance / 0.11);
                    return(<span className={fontWeights[index]}>{tokenImp.word} </span>);
                })}
            </CardContent>
          </div>
        </Card>
      </div>
    );
}

export default ExplainedImage;