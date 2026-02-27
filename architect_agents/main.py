from pipeline import ArchitecturePipeline

if __name__ == "__main__":
    metadata = {
        "style": "gothic",
        "period": "medieval",
        "country_normalized": "France",
        "primary_materials": "stone",
        "purpose": "cathedral"
    }

    # sketch_path = "data/images/000000.jpg"
    conditioning_path = "/content/drive/MyDrive/validation/validation/conditioning_images/000029.jpg"

    pipe = ArchitecturePipeline()
    image, prompt, score = pipe.run(
        metadata,
        # sketch_path,
        conditioning_path,
        iterations=3
    )

    image.save("/content/agent-project/architect_agents/results.png")

    print("\nFinal prompt:")
    print(prompt)

    print("Final score:", score)

