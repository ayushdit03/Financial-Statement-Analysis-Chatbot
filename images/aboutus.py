import streamlit as st

def about_us():
    # List of images and their descriptions
    content = [
        {
            "image": "E:\cohnizant\main2\main2\images\img2.png",
            "title": "Ayush Jain",
            "description": "We strive to innovate and create solutions that make a positive impact on the world."
        },
        {
            "image": "E:\cohnizant\main2\main2\images\img3.png",
            "title": "Dhruv Yaranalkar",
            "description": "A diverse group of passionate individuals working together to achieve our goals."
        },
        {
            "image": "E:\cohnizant\main2\main2\images\img4.png",
            "title": "Nikhil Ingale",
            "description": "Integrity, innovation, and customer satisfaction are at the core of everything we do."
        },
        {
            "image": "E:\cohnizant\main2\main2\images\img5.png",
            "title": "Atharva Pimple",
            "description": "Founded in 2010, we've grown from a small startup to a global leader in our industry."
        },
        {
            "image": "E:\cohnizant\main2\main2\images\img1.jpg",
            "title": "Sakshi",
            "description": "We're proud of our numerous awards and recognitions for excellence in our field."
        },
        {
            "image": "E:\cohnizant\main2\main2\images\img1.jpg",
            "title": "Srushti Rajput",
            "description": "We're constantly evolving and looking forward to the exciting challenges ahead."
        }
    ]

    for i, item in enumerate(content):
        col1, col2 = st.columns(2)
        
        if i % 2 == 0:  # Even index, image on the left
            with col1:
                st.image(item["image"],width=250)
            with col2:
                st.subheader(item["title"])
                st.write(item["description"])
        else:  # Odd index, image on the right
            with col1:
                st.subheader(item["title"])
                st.write(item["description"])
            with col2:
                st.image(item["image"],width=250)
        
        st.write("---")  # Add a separator between sections

if __name__ == "__main__":
    about_us()