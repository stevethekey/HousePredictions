import sklearn

print(sklearn.__version__)  # checks to see if you guys have scikit-learn version.
# Should print out 0.22 or higher, like 1.something

if __name__ == "__main__":
    house = input("Enter a house name: ")
    print(house, "is worth $100,000")
