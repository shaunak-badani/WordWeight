import { Card, CardContent } from "@/components/ui/card";

const ExplainedImage = (props: any) => {
    const {image, key} = props;

    const imageSrc = image.masked_image;
    return (
        
        <div className="flex justify-center m-6 sm:m-6" key={key}>
        <Card className="w-3/4 flex">
          <img 
            src={`data:image/jpeg;base64,${imageSrc}`} 
            alt="Business"
            style={{ maxWidth: "40%", height: "auto" }} 
            />
          <div className="w-full flex flex-row justify-center">
            <CardContent>
              <div className="flex flex-row justify-evenly">
                  <div className="flex flex-row w-full items-center gap-4 justify-center">
                      <div className="flex flex-col space-y-1.5">
                          <div>Test</div>
                      </div>
                  </div>
              </div>
            </CardContent>
          </div>
        </Card>
      </div>
    );
}

export default ExplainedImage;