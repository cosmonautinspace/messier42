from PIL import Image


inputName = input("Enter the name of the input image to crop >> ")
targetName = input("Enter the name of the target image to crop >> ")

print("(x1,x2)__________")
print("|                |\n|                |\n|                |")
print("___________(y1,y2)")

x1 = int(input("Enter a value for x1 >> "))
x2 = int(input("Enter a value for x2 >> "))
y1 = int(input("Enter a value for y1 >> "))
y2 = int(input("Enter a value for y2 >> "))

inputImage = Image.open(f"data/images/input/{inputName}")
targetImage = Image.open(f"data/images/target/{targetName}")

inputImage = inputImage.crop((x1,x2,y1,y2))
targetImage = targetImage.crop((x1,x2,y1,y2))

inputImage.save("data/images/input/croppedInput.TIF")
targetImage.save("data/images/target/croppedTarget.TIF")