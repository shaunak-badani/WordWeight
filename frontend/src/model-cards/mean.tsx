import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import BackdropWithSpinner from "@/components/ui/backdropwithspinner";
import backendClient from "@/backendClient";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import ImageOverlay from "./image-wrapper";


const Mean = () => {

    const [isLoading, setLoading] = useState(false);
    const [query, setQuery] = useState("");
    const [response, setResponse] = useState("");

    const presetValues = [
        "A futuristic cityscape",
        "Jumbo shrimp next to a tiny house"
    ]

    const handlePromptInput = async(query: string) => {
        setLoading(true);
        const response = await backendClient.get("/generate", {
            params: {
                prompt: query
            }
        });
        if(response.data)
        {
            setResponse(response.data.image_base64);
        }
        setLoading(false);
    }

    return (
        <>
            <h6 className="pb-6 sm:pb-6 text-xl">Write a prompt to generate an image</h6>
            <Textarea
                value={query}
                onChange = {(e) => setQuery(e.target.value)} 
                placeholder="Enter your query here!" />
            <div className="my-6">
                <Select onValueChange={setQuery}>
                    <SelectTrigger id="querySelector">
                    <SelectValue placeholder="Choose a sample query from the presets.." />
                    </SelectTrigger>
                    <SelectContent position="popper">
                    {presetValues.map((preset, index) => 
                        (<SelectItem
                            key={index}
                            value={preset} 
                            >
                            {preset}
                        </SelectItem>)
                    )}
                    </SelectContent>
                </Select>
            </div>
            <Button className="p-6 sm:p-6 rounded-2xl m-8 sm:m-8" onClick={() => handlePromptInput(query)}>
                Generate
            </Button>
            {response.length > 0 && <ImageOverlay
                image={response}
                prompt={prompt}
                />}
            {isLoading && <BackdropWithSpinner />}
        </>
    )
};


export default Mean;