import streamlit as st

def show():
    # Custom CSS for dark theme, cyan accents, footer styles, and Bootstrap
    st.markdown("""
    <style>
    .cyan-text {
        color: #00FFFF;
    }
    .service-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .footer {
        background-color: #262730;
        color: #ffffff;
        padding: 20px;
    }
    .footer a {
        color: #00FFFF;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .social-icons a {
        color: #ffffff;
        margin-right: 15px;
        font-size: 24px;
    }
    .social-icons a:hover {
        color: #00FFFF;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1 style='text-align: center;'><span class='cyan-text'>PIXELS</span></h1>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.markdown("<h2>WE ARE <span class='cyan-text'>CREATIVE</span> DESIGNERS</h2>", unsafe_allow_html=True)
        st.write("Lorem ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.")
    
    with col2:
        st.image("https://via.placeholder.com/300x300", use_column_width=True)
    
    # Services section
    st.markdown("<h2>WHAT WE <span class='cyan-text'>DO?</span></h2>", unsafe_allow_html=True)
    st.write("Lorem ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='service-box'>
        <h3>Website Design</h3>
        <p>We can design for you a website and we can upload them.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='service-box'>
        <h3>Mobile & Desktop App</h3>
        <p>We can create for you mobile and desktop app.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='service-box'>
        <h3>UI & UX Design</h3>
        <p>We can create for you mobile and desktop app.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='service-box'>
        <h3>Editing Photo</h3>
        <p>We can design for you a website and we can upload them.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About Us section
    st.markdown("<h2>WHO ARE <span class='cyan-text'>WE?</span></h2>", unsafe_allow_html=True)
    st.write("Lorem ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='service-box'>
        <h3>Clean Code</h3>
        <p>Lorem ipsum is simply dummy text of the printing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='service-box'>
        <h3>Modern Design</h3>
        <p>Lorem ipsum is simply dummy text of the printing.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PROJECT", "300+")
    col2.metric("PLEASURE", "8,9")
    col3.metric("CUSTOMER", "3000+")
    col4.metric("TEAM MEMBERS", "23")

    # Footer
    st.markdown("""
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-lg-3 col-md-6">
                    <h3>Get in Touch</h3>
                    <p>Don't miss any updates of our new templates and extensions!</p>
                    <form action="#" method="post" novalidate="true">
                        <input type="text" name="EMAIL" class="form-control" placeholder="Email">
                        <button class="btn btn-primary" type="submit">Subscribe</button>
                    </form>
                </div>
                <div class="col-lg-3 col-md-6">
                    <h3>Connect To Team</h3>
                    <ul>
                        <li><a href="https://www.linkedin.com/in/meet-agarwal-720160228/">Meet Agarwal</a></li>
                        <li><a href="https://www.linkedin.com/in/mohit-khandelwal-034aaa236/">Mohit Khandelwal</a></li>
                        <li><a href="https://www.linkedin.com/in/vaibhavi-dixit/">Vaibhavi Dixit</a></li>
                        <li><a href="https://www.linkedin.com/in/ayush-jain-8b6985231">Ayush Jain</a></li>
                    </ul>
                </div>
                <div class="col-lg-3 col-md-6">
                    <h3>Help</h3>
                    <ul>
                        <li><a href="#">FAQ</a></li>
                        <li><a href="#">Terms & Conditions</a></li>
                        <li><a href="#">Support Policy</a></li>
                        <li><a href="#">Privacy</a></li>
                    </ul>
                </div>
                <div class="col-lg-3 col-md-6">
                    <h3>Socials</h3>
                    <div class="social-icons">
                        <a href="https://www.facebook.com/" class="fab fa-facebook"></a>
                        <a href="https://twitter.com/" class="fab fa-twitter"></a>
                        <a href="https://www.linkedin.com/" class="fab fa-linkedin"></a>
                        <a href="https://in.pinterest.com/" class="fab fa-pinterest"></a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <div class="row align-items-center">
                    <div class="col-lg-6 col-sm-7">
                        <p class="mb-0">© mansionify Inc. 2024 All rights reserved.</p>
                    </div>
                    <div class="col-lg-6 col-sm-5 text-right">
                        <p>Made with &hearts; by <a href="/home" target="_blank">Mansionify</a></p>
                    </div>
                </div>
            </div>
        </div>
    </footer>
    """, unsafe_allow_html=True)
