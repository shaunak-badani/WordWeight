import Information from '../components/information'
import Mean from '../model-cards/mean'
import { Card } from "@/components/ui/card"



function Home() {
    return (
        <div>
              <h1 className="scroll-m-20 tracking-tight lg:text-3xl">
                See What Words Create — and Why.
              </h1>
              <p className="leading-7 [&:not(:first-child)]:mt-6 m-6 sm:m-6">
                WordWeight lets you generate stunning images from text using cutting-edge diffusion models — 
                then pulls back the curtain to reveal which words carried the most weight. 
                Visualize token influence, explore semantic relevance, and even crop regions of your 
                image to trace back the words that brought them to life.
              </p>
              <Information />
                <Card className="p-20">
                  <Mean />
                </Card>
            </div>
    );
  }
  
  export default Home;