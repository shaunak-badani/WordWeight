import { useEffect, useState } from "react";
import backendClient from "@/backendClient";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import useGlobalStore from "@/store/store";
import ExplainedImage from "./ExplainedImage";

const Examples = () => {
  const [isLoading, setLoading] = useState(false);
  const [explainedImages, setExplainedImages] = useState([]);

  const fetchImages = async() => {
    setLoading(true);
    const response = await backendClient.get("/explained-images");
    if(response.data)
    {
      setExplainedImages(response.data)
    }
    setLoading(false);
  }
  const error = useGlobalStore(state => state.error);

  useEffect(() => {
    fetchImages();
  }, []);

  return (
    <>
      <h1>Previously run images</h1>
      {!error && explainedImages.map((image, index) => 
        <ExplainedImage
          key={index}
          image={image}
          />
      )}
      {isLoading && <BackdropWithSpinner />}
    </>
  );
}

export default Examples;