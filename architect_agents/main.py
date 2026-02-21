from pipeline import ArchitecturePipeline

if __name__ == "__main__":
    metadata = {
        "style": "gothic",
        "period": "medieval",
        "country_normalized": "France",
        "primary_materials": "stone",
        "purpose": "cathedral"
    }

    sketch_path = "data/images/000000.jpg"
    conditioning_path = "data/conditioning_images/000000.jpg"

    pipe = ArchitecturePipeline()
    image, prompt, score = pipe.run(
        metadata,
        sketch_path,
        conditioning_path,
        iterations=3
    )

    image.save("results/final_architecture.png")

    print("\nFinal prompt:")
    print(prompt)
    print("Final score:", score)