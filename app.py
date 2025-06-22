import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import requests
import re
import streamlit as st
from datetime import datetime
import uuid
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import io
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import time
import tempfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL_CONFIG = {
    "openai": {
        "enabled": True,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "/chat/completions",
    },
    "grok": {
        "enabled": True,
        "api_key": os.getenv("XAI_API_KEY"),
        "model_name": "grok-3",
        "base_url": "https://api.x.ai/v1",
        "chat_endpoint": "/chat/completions",
        "fallback_models": [
            "grok-3-fast",
            "grok-3-mini",
        ],
    },
    "gemini": {
        "enabled": True,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_name": "models/gemini-1.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "chat_endpoint": ":generateContent",
    },
    "local": {
        "enabled": True,
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "base_url": None,
        "chat_endpoint": None,
        "api_key": None,
    },
}
DATA_FILE = "personal_data.json"

# Sample personal data (unchanged)
personal_data = {
    "profile": {
        "name": "Sumanth Bejugam",
        "email": "sumanthbejugam@gmail.com",
        "phone": "5714650804",
        "linkedin": "https://linkedin.com/in/bejugamsumanth",
        "summary": "Full Stack Software Developer with 4+ years of experience building scalable web, mobile, and IoT applications using Flutter, React, Kotlin, Node.js, and Python. Proven track record of delivering production-ready solutions under tight deadlines, with 50,000+ app downloads, enterprise adoption, and government recognition. Founder of a tech startup and winner of multiple hackathons. Skilled in cloud automation (AWS, Azure), CI/CD, ML, and BLE-based IoT systems. Known for rapid learning, clean architecture, and cross-platform expertise. Eligible for Public Trust Clearance.",
    },
    "experiences": [
        {
            "role": "Full Stack Developer",
            "company": "Epitome Research And Innovation Inc, Herndon, VA",
            "duration": "Sep 2024 – Present",
            "description": [
                "Developed and deployed a super app using React, Django, and PostgreSQL, streamlining operations for 50+ clients.",
                "Integrated QuickBooks API for seamless financial data synchronization, improving accounting efficiency by 25% for end users.",
                "Engineered a responsive web application using React with Next.js and Mantine, reducing page load times from 5 seconds to 3 seconds.",
                "Optimized user-facing interfaces for simulation management in a high-security environment, reducing database interaction time from 1 second to 0.6 seconds.",
                "Designed and implemented RESTful APIs with Flask, reducing development cycles from 20 weeks to 15 weeks.",
                "Implemented 3D object visualization in React using Three.js, cutting STL file rendering time from 10 seconds to 5 seconds.",
                "Designed a scalable MongoDB data storage solution, reducing query latency from 200 ms to 130 ms.",
                "Orchestrated API and website hosting on AWS EC2 instances, reducing server response times from 500 ms to 325 ms.",
                "Automated AWS infrastructure management using Boto3 for EC2, Elastic File System (EFS), and Elastic IP (EIP), reducing monthly costs from by 300.",
                "Established CI/CD pipelines with AWS CodeBuild and CodePipeline, reducing build times from 20 minutes to 12 minutes.",
                "Developed a high-performance cross-platform mobile application using Flutter for web, Android, and iOS, cutting development time from 10 weeks to 6 weeks.",
                "Engineered native Android applications using Java and Kotlin with Jetpack libraries, implementing advanced features such as background services and location-based triggers.",
                "Enhanced application reliability, reducing failure rate from 10% to 8% through comprehensive testing, logging, and performance optimization.",
                "Leveraged advanced Linux commands and open-source machine learning libraries to support underwater simulation workflows in secure environments.",
                "Collaborated with data scientists, system architects, and project leads in a cross-functional team to deliver solutions meeting stringent requirements.",
                "Maintained strict adherence to security protocols and confidentiality guidelines for sensitive data.",
                "Developed a custom Bluetooth Low Energy (BLE) Python package for Windows, enabling seamless communication between a Flutter Android app and a Windows Python script, reducing connection latency from 500 ms to 200 ms.",
                "Implemented BLE and WebSocket communication protocols to facilitate real-time data exchange between an Android app and a robotic eye via a Windows intermediary, achieving 99% reliability in command transmission.",
                "Engineered a responsive web application using React with Next.js and Mantine, reducing page load times from 5 seconds to 3 seconds.",
            ],
        },
        {
            "role": "Associate Developer Intern (Boomi Integration Architect)",
            "company": "Quotient Technologies Inc., Bengaluru, India",
            "duration": "Dec 2021 – June 2022",
            "description": [
                "Formulated seamless Dell Boomi integration flows to synchronize data between Salesforce and NetSuite, achieving a 100% success rate.",
                "Crafted efficient integration flows for REST Endpoint and Employee Off-boarding in Okta, reducing process time from 10 minutes to 1 minute.",
                "Designed and implemented sophisticated integration solutions with Dell Boomi, streamlining complex business processes.",
                "Implemented data pipeline automation using Apache Airflow, reducing manual intervention from 5 hours to 1 hour weekly.",
                "Developed and optimized data integration and transformation processes using Snowflake, enhancing storage efficiency.",
                "Built a web application using EJS and Express with MVC architecture, shortening development cycles from 10 weeks to 7 weeks.",
            ],
        },
        {
            "role": "Founder & Lead Software Developer",
            "company": "Confegure Techsols Pvt Ltd, Hyderabad, India",
            "duration": "May 2020 – May 2024",
            "description": [
                "Formulated the company’s 5-year vision, goals, and objectives, crafting a strategic roadmap aligned with market trends.",
                "Interviewed, onboarded, and mentored over 10 technical and business professionals, fostering a high-performing team culture.",
                "Analyzed market and industry trends to develop actionable strategic plans.",
                "Led multiple projects from conception to publication using Agile methodologies, balancing independent work with effective team collaboration.",
                "Developed diverse web applications using MERN, React with Firebase, Flutter for web, and Java Spring Boot with React/Angular, reducing load times from 4 seconds to 2.8 seconds.",
                "Built and published mobile applications using React Native, Java, Kotlin, and Flutter, achieving over 20,000 downloads for an AI app.",
                "Designed scalable backends and databases with Node.js, Express, Java Spring Boot, SQL, and MongoDB, reducing API response times from 500 ms to 300 ms.",
                "Engineered an ESP32-based IoT module using Arduino and MQTT in C++, enabling AI-driven smart home automation.",
                "Built and programmed a sophisticated marine research tool, improving plankton imaging and analysis accuracy from 60% to 78%.",
                "Implemented Firebase services (Auth, Firestore, Analytics, Crashlytics), reducing app crash rates from 5% to 3%.",
                "Created wireframes and prototypes in Figma, shortening design iteration time from 4 weeks to 3 weeks.",
                "Conducted comprehensive testing and quality assurance, achieving a 95% bug identification and resolution rate.",
                "Enhanced a Python package by resolving 95% of critical bugs, improving its reliability.",
            ],
        },
    ],
    "projects": [
        {
            "name": "Lock Down Mart",
            "description": "Lock Down Mart is a comprehensive e-commerce solution designed to help local grocery stores operate efficiently during the COVID-19 lockdown. The project consists of two apps—one for grocery store owners and the other for customers. The store owner app allows for full product management, including the addition, editing, deletion, and management of offers and categories. The customer app provides an easy-to-use platform for customers to search, order, and track products in real-time. A virtual queue system is implemented to ensure safe, socially-distanced shopping by limiting the number of customers in the store at any given time. The app integrates with Google Maps for location-based features, allowing users to find nearby stores. The app supports multiple login methods including phone number, email, Google, and Facebook. Customers receive notifications from store owners regarding product availability and offers. The project is developed with a focus on simplicity, safety, and convenience, ensuring a seamless shopping experience while maintaining COVID-19 safety measures. The project was recognized by the IT Ministry of Telangana and published in a magazine for its innovative approach to solving problems faced by local businesses during the pandemic.",
            "technologies": [
                "Flutter",
                "Dart",
                "Google Maps API",
                "Firebase Authentication",
                "Firestore",
                "Firebase Storage",
                "Figma",
                "Adobe XD",
                "HTML",
                "CSS",
                "Bootstrap",
                "NodeJS",
                "ExpressJS",
            ],
        },
        {
            "name": "Udhyogulu",
            "description": "Udhyogulu is a cross-platform news application designed to deliver employee-focused news tailored to their interests and locations. The app allows publishers to manage and update news content, including adding, editing, and deleting news articles, roles, and headlines. Users can access news categorized by district, state, and topic, with the ability to filter content according to their preferences. The app integrates with Google Maps for location-based news delivery and enables users to subscribe to specific topic notifications. Firebase Messaging is used for real-time alerts, ensuring that users stay up-to-date with relevant news. The project includes an Android app, an iOS app, and a web-based interface, with seamless integration between them. A robust backend powered by PHP handles content management, and AWS S3 is used for scalable image storage.",
            "technologies": [
                "React",
                "Redux",
                "Firebase Messaging",
                "HTML",
                "CSS",
                "Bootstrap",
                "PHP",
                "AWS EC2",
                "AWS S3",
            ],
        },
        {
            "name": "Cinemawala",
            "description": "Cinemawala is a comprehensive cross-platform application designed to manage all aspects of film production, including cast, crew, scenes, props, budget, costume, schedule, and locations. It allows users to create projects, assign roles, and manage permissions for each category such as scenes, props, and schedules. Users can add images to each category, generate detailed PDF reports for each category and the entire project, and have an optimized, responsive UI for all devices. The app includes a flexible role hierarchy, calendar UI for scheduling, and utilizes cookie-based authentication for secure access. Built using Flutter for cross-platform development, NodeJS for backend services, MongoDB for database management, and Firebase for messaging.",
            "technologies": [
                "Android Java",
                "Swift",
                "XML",
                "Flutter",
                "Dart",
                "JS",
                "Firebase Messaging",
                "Node JS",
                "Express JS",
                "MongoDB",
                "Cookie Authentication",
            ],
        },
        {
            "name": "Confegure Website",
            "description": "Confegure Website is a responsive and dynamic platform developed to showcase the portfolio of a startup company. The website offers seamless user experience (UX) across all devices and integrates social media for enhanced engagement. It includes features like web analytics and FAQs to provide valuable insights into user behavior and to address customer inquiries efficiently. The platform was initially built using Flutter, which was later migrated to React to improve performance and scalability. The modern and flexible design is optimized for various screen sizes, ensuring accessibility and smooth navigation. Additionally, CSS was extensively customized to match the brand's visual identity, creating a unique, professional appearance.",
            "technologies": ["React", "React Native", "AWS EC2", "CSS"],
        },
        {
            "name": "Oil Seed and Pests Hub",
            "description": "Oil Seed and Pests Hub is a comprehensive web application developed for the Telangana State Government Oil Seed and Pests Research Organization. This platform allows users to upload and view detailed information about oil seeds and pests affecting oil plants. It includes the ability to upload pest descriptions, including causes and pesticide recommendations, and oil seed details for research purposes. The website is designed to be secure, ensuring that all data can be accessed and managed efficiently from anywhere. The system also integrates a file conversion module to handle data storage effectively, making the management of complex data easy and organized.",
            "technologies": ["Flutter", "PHP", "Linux"],
        },
        {
            "name": "Tic Trac",
            "description": "Tic Trac is the world’s first Movie Bookings Tracking Application, designed to notify users as soon as bookings open for a selected movie in a selected theatre on a specified date. The app supports login via mobile number with 249 country codes and tracks bookings for over 500,000 screens across 29 states, 10 major cities, and 1,800+ regions. It runs seamlessly in the background, even when the app is closed or not actively in use, ensuring that users never miss the start of a booking. Additionally, Tic Trac integrates with Twitter, automatically tweeting about booking openings, and can handle large-scale server requests effortlessly. It also features Firebase Authentication, real-time notifications, and a sophisticated database system built using MongoDB. The app has garnered over 15,000 downloads, receiving positive feedback from users and securing a project from one of the largest promotion agencies in India.",
            "technologies": [
                "Flutter",
                "Android Java",
                "XML",
                "NodeJS",
                "ExpressJS",
                "Firebase Authentication",
                "MongoDB",
                "Mongoose",
                "REST",
                "Firebase Analytics",
                "Firebase Messaging",
                "Firebase In-App Messaging",
                "Figma",
                "Adobe XD",
                "Web Scraping",
                "UI Design",
                "UX Design",
            ],
        },
        {
            "name": "Ombi Non Medical Transport Service",
            "description": "A responsive website developed for an ambulance service company that showcases all the services provided by the company and includes a contact form for users to inquire further. The website is designed to be user-friendly, providing a seamless experience across all devices. The backend is developed using Javascript, and the contact form is integrated with a third-party module to facilitate inquiries from potential customers. The project emphasizes a clean and professional design, ensuring that all information is easily accessible to users seeking non-medical transport services.",
            "technologies": [
                "HTML",
                "CSS",
                "Bootstrap",
                "Javascript",
                "Figma",
                "Adobe XD",
                "Web Scraping",
                "UI Design",
                "UX Design",
            ],
        },
        {
            "name": "Red and Blue Appliance Repair",
            "description": "Developed a responsive and SEO-optimized web application for an appliance repair service using ASP.NET MVC and C#. The application showcased services and enabled user inquiries via a contact form integrated with backend validation and email notification. Designed for optimal performance and lead conversion with modern UI/UX principles.",
            "technologies": [
                "ASP.NET MVC",
                "C#",
                "Entity Framework",
                "SQL Server",
                "HTML",
                "CSS",
                "Bootstrap",
                "JavaScript",
                "Razor Pages",
                "Google SEO Tools",
            ],
        },
        {
            "name": "Laptop Tracking/Localization",
            "description": "This is a Python module designed to track and localize laptops within a building using Wi-Fi network signal strengths. The system leverages data from Wi-Fi signals to estimate the laptop's position with a minimal error margin of 5 meters. The project involves the design and implementation of a location collection protocol, data gathering through multiple iterations, and cross-validation using machine learning methods to enhance the accuracy of the location estimates. The system is optimized for minimal error and reliable performance in indoor environments.",
            "technologies": ["Python", "Pandas", "Numpy", "Pywifi"],
        },
        {
            "name": "Intelliswitch",
            "description": "This is a smart home solution with IoT capabilities, designed to control and monitor home appliances using a mobile app over the cloud from anywhere in the world from mobile. The system utilizes an ESP32 microcontroller for Bluetooth Low Energy (BLE) communication, enabling users to manage devices remotely. The project includes a custom MQTT server that facilitates seamless communication between the mobile app and the ESP32 module. The solution is designed to be user-friendly, with a focus on security and reliability, making it suitable for various home automation applications. Mobile app is made with Flutter, and the backend is developed using NodeJS and ExpressJS. Programming of the ESP32 module is done using Arduino C, and the MQTT server is implemented using Mosquitto.",
            "technologies": [
                "Flutter",
                "BLE",
                "MQTT",
                "Mosquitto",
                "Arduino C",
                "NodeJS",
                "ExpressJS",
            ],
        },
        {
            "name": "Reach Alert",
            "description": "Reach Alert is an Android application that provides a location-triggered alarm system. Users can set a specific location and radius, and the app will notify them with an alarm when they enter the defined area. The app integrates Google Maps for precise location pinning, viewing, and manipulation, displaying key details such as name, address, latitude/longitude, and distance. It tracks the user's location continuously, whether the app is open, in the background, or removed from the background. The app supports various login methods, including phone number, email, Google, and Facebook. The radius can be adjusted between 500 meters and 5 kilometers, making the app highly customizable for different needs. Firebase Authentication and Firestore are used for secure user authentication and data storage, while Google Ads and Unity Ads are embedded for monetization. The app has gained significant traction, with over 2000 downloads, including 100+ downloads in the first month.",
            "technologies": [
                "Android Java",
                "XML",
                "Flutter",
                "Dart",
                "Google Maps API",
                "Firebase Authentication",
                "Firestore",
                "Google Ads",
                "Unity Ads",
            ],
        },
        {
            "name": "NLPositionality",
            "description": "Investigating biases in NLP models and datasets through multilingual and robustness analyses on social acceptability and hate speech tasks.",
            "technologies": [
                "Python",
                "Jupyter Notebook",
                "googletrans",
                "nlpaug",
                "google-api-python-client",
                "openai",
            ],
        },
    ],
    "skills": [
        "Full Stack Development",
        "Frontend Development",
        "Backend Development",
        "Mobile App Development",
        "Cross-Platform App Development",
        "IoT System Design",
        "System Architecture",
        "Agile Development",
        "Project Leadership",
        "Wireframing & Prototyping",
        "UI/UX Design",
        "Technical Mentorship",
        "Testing & Debugging",
        "Web Scraping & Automation",
        "REST API Development",
        "Bluetooth Low Energy (BLE) Communication",
        "Cloud Infrastructure Automation",
        "Database Design & Optimization",
        "DevOps Pipelines",
        "ETL Pipelines",
        "Rapid Prototyping",
        "Performance Optimization",
        "Security Integration",
        "Speech & Location Detection",
        "Real-Time Data Communication",
        "HTML",
        "CSS",
        "JavaScript",
        "TypeScript",
        "Dart",
        "Java",
        "Kotlin",
        "Python",
        "C++",
        "PHP",
        "SQL",
        "Bash",
        "Arduino C",
        "Flutter",
        "Jetpack Compose",
        "Android SDK",
        "ReactJS",
        "React Native",
        "Redux",
        "Node.js",
        "Express.js",
        "Spring Boot",
        "Django",
        "Next.js",
        "Bootstrap",
        "Tailwind CSS",
        "Firebase",
        "Firestore",
        "Firebase Auth",
        "Firebase Messaging",
        "Firebase Analytics",
        "MongoDB",
        "MySQL",
        "PostgreSQL",
        "Room DB",
        "Snowflake",
        "Docker",
        "Jenkins",
        "Fastlane",
        "Boomi",
        "Airflow",
        "Git",
        "GitHub",
        "Figma",
        "Adobe XD",
        "Puppeteer",
        "Selenium",
        "Chrome DevTools",
        "Framer Motion",
        "Stripe",
        "Razorpay",
        "AWS EC2",
        "AWS S3",
        "AWS CloudFront",
        "AWS CodeBuild",
        "Google Cloud Platform",
        "Azure VM",
        "Azure SQL Server",
        "Azure Blob Storage",
        "GitHub Actions",
        "Certbot",
        "Nginx",
        "PM2",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Matplotlib",
        "PyWiFi",
        "NLP",
        "LLMs",
        "Data Visualization",
        "Data Cleaning",
        "ESP32",
        "Arduino",
        "MQTT",
        "Mosquitto",
        "GATT Server",
        "Raspberry Pi",
        "Wi-Fi Signal Localization",
    ],
    "achievements": [
        "Co-founded a startup and maintained it for 4 years, serving over 15 clients and 5 products.",
        "Won 2 hackathons for innovative web solutions, demonstrating creativity and technical prowess.",
        "Co-authored a research paper on machine learning for data generation, published.",
        "Led the development of an Android app that achieved over 20,000 downloads within 6 months and maintained a 5-star rating for 4 consecutive years.",
    ],
    "certfications": [
        {
            "name": "Boomi - Associate Developer",
            "description": "Specialized in Salesforce and NetSuite integration.",
        },
        {
            "name": "NASSCOM AI",
            "description": "Gained foundational AI and machine learning skills.",
        },
        {
            "name": "Introduction to Cybersecurity",
            "description": "Acquired basic knowledge of cybersecurity principles and practices.",
        },
        {
            "name": "Certified Cloud Security Professional (CCSP)",
            "description": "Demonstrated expertise in cloud security architecture, design, operations, and service orchestration.",
        },
        {
            "name": "Certified Information Security Manager (CISM)",
            "description": "Proficient in managing, designing, and assessing an enterprise's information security.",
        },
        {
            "name": "Certified Information Security Manager (CISM)",
            "description": "Proficient in managing, designing, and assessing an enterprise's information security.",
        },
        {
            "name": "Certified Information Systems Security Professional (CISSP)",
            "description": "Mastered the eight domains of information security, including security and risk management, asset security, and more.",
        },
        {
            "name": "CompTIA Security+ 501",
            "description": "Gained foundational skills in cybersecurity, including threat management, cryptography, and identity management.",
        },
        {
            "name": "Problem Solving Certificate",
            "description": "Developed strong analytical and problem-solving skills applicable to various technical challenges.",
        },
    ],
}


def get_google_creds(SCOPES):
    """
    Returns a google.oauth2.service_account.Credentials object for the given scopes.
    - If ENVIRONMENT=local, loads from 'service_account.json'.
    - Else, loads from SERVICE_ACCOUNT_JSON env var (the full JSON as a string).
    """
    from google.oauth2.service_account import Credentials
    import tempfile
    import json

    environment = os.getenv("ENVIRONMENT", "local").lower()
    if environment == "local":
        creds_path = "service_account.json"
        if not os.path.exists(creds_path):
            raise RuntimeError(f"Service account file '{creds_path}' not found!")
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    else:
        sa_json = os.getenv("SERVICE_ACCOUNT_JSON")
        if not sa_json:
            raise RuntimeError("SERVICE_ACCOUNT_JSON environment variable not set!")
        # Write to a temp file for compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            tmp.write(sa_json.encode())
            tmp.flush()
            creds = Credentials.from_service_account_file(tmp.name, scopes=SCOPES)
    return creds


# Save personal data to JSON
def save_personal_data():
    with open(DATA_FILE, "w") as f:
        json.dump(personal_data, f, indent=2)


# Prepare dataset for fine-tuning
def prepare_dataset():
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    texts = []
    for section, content in data.items():
        if section == "profile":
            texts.append(f"Profile: {content['summary']}")
        elif section == "experiences":
            for exp in content:
                texts.append(
                    f"Experience: {exp['role']} at {exp['company']} ({exp['duration']}) - {', '.join(exp['description'])}"
                )
        elif section == "projects":
            for proj in content:
                texts.append(
                    f"Project: {proj['name']} - {proj['description']} Technologies: {', '.join(proj['technologies'])}"
                )
        elif section == "skills":
            texts.append(f"Skills: {', '.join(content)}")
        elif section == "achievements":
            for ach in content:
                texts.append(f"Achievement: {ach}")

    dataset = Dataset.from_dict({"text": texts})
    return dataset


# Generic model class for API-based LLMs
class LLMModel:
    def __init__(self, model_config):
        self.model_name = model_config["model_name"]
        self.api_key = model_config["api_key"]
        self.base_url = model_config["base_url"]
        self.chat_endpoint = model_config["chat_endpoint"]
        self.fallback_models = model_config.get("fallback_models", [])
        self.enabled = model_config["enabled"]
        self.is_local = model_config.get("base_url") is None

        if self.is_local:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
            self.model = AutoModelForCausalLM.from_pretrained(model_config["model_name"])
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def run_sync(self, prompt):
        if self.is_local:
            result = self.pipe(prompt, max_new_tokens=512, temperature=0.7)
            return result[0]["generated_text"]
        
        if not self.enabled:
            raise Exception(f"{self.model_name} is disabled in MODEL_CONFIG")
        
        if not self.api_key:
            raise Exception(
                f"Missing API key for {self.model_name}. Check .env file or environment variables."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1500,
            "temperature": 0.7,
        }
        api_url = f"{self.base_url}{self.chat_endpoint}"
        logger.debug(f"Attempting API call to {api_url} with model {self.model_name}")

        models_to_try = [self.model_name] + self.fallback_models
        for model in models_to_try:
            if self.model_name == 'gemini':
                api_url = f"{self.base_url}/{model}{self.chat_endpoint}"
            payload["model"] = model
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                logger.debug(f"API response: {json.dumps(result, indent=2)}")
                return content
            except requests.HTTPError as e:
                logger.error(f"HTTP error for {model}: {str(e)}")
                if e.response.status_code == 404:
                    logger.error(
                        f"API endpoint not found for {model}. URL: {api_url}, Response: {e.response.text}"
                    )
                    if model == models_to_try[-1]:
                        raise Exception(
                            f"API endpoint not found for any model. Last tried: {model}, URL: {api_url}, Response: {e.response.text}"
                        ) from e
                else:
                    raise Exception(f"API error for {model}: {str(e)}") from e
            except requests.RequestException as e:
                logger.error(f"Request error for {model}: {str(e)}")
                raise Exception(f"API error for {model}: {str(e)}") from e


# Initialize model agent
def initialize_agent():
    enabled_models = [
        name for name, config in MODEL_CONFIG.items() if config["enabled"]
    ]
    if not enabled_models:
        raise Exception("No models enabled in MODEL_CONFIG")

    model_name = enabled_models[0]
    print(f"Using model: {model_name}")
    return LLMModel(MODEL_CONFIG[model_name])


# Updated generate_document function
def generate_document(job_description, document_type, agent):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    if document_type == "Resume":
        # Removed this from prompt as it is longing the prompt thereby using more tokens
        # Web Search Results: {web_search_results}
        # "overview": "A 2-3 sentence summary highlighting my most relevant skills, experiences, and achievements based on the job description. Use insights from the web search results to align with the company’s focus or job requirements.",
        # "achievements": [
        #     "Notable achievement 1",
        #     "Notable achievement 2",
        # ]
        prompt = f"""
        Here is my personal portfolio data:
        Profile: {data['profile']['summary']}
        Experiences: {json.dumps(data['experiences'], indent=2)}
        Projects: {json.dumps(data['projects'], indent=2)}
        Skills: {', '.join(data['skills'])}
        Achievements: {json.dumps(data['achievements'], indent=2)}

        Job Description: {job_description}

        Generate a JSON object for a tailored resume with the following structure:
        {{
            "work_experience": [
                {{
                    "role": "Role title",
                    "company": "Company name",
                    "duration": "Duration",
                    "bullets": [
                        "Relevant bullet point, highlighting a key achievement or responsibility in eact technology or other technology thats similar to requirement.",
                        "another point",
                        "...up to 9 points, at least 4, using absolute metrics or concrete numbers when possible."
                    ]
                }}
                // ...repeat for 3 companies' experiences
            ],
            "projects": [
                {{
                    "name": "Project Name",
                    "description": "Brief description (1 sentence) highlighting relevance to the job.",
                    "technologies": ["Tech1", "Tech2", ...]
                }}
                // ...repeat for 3-4 most relevant projects
            ],
            "skills": {{
                "Category1": ["Skill1", "Skill2", ...],
                "Category2": ["Skill3", "Skill4", ...]
                // ...categorize and list all relevant skills
            }},
            "company_name": "Extract the company name from the job description",
            "role_name": "Extract the role name from the job description",
        }}
        
        Additional Instructions:
        - Only output valid JSON, no markdown or extra text.
        - Prioritize content that directly matches the job’s key requirements (skills, technologies, responsibilities).
        - Use web search results to tailor the overview and experience descriptions to the company’s values or additional job context.
        - Keep content concise, professional, and impactful, limiting bullet points to 2-7 per experience.
        - Limit the resume to 1-2 pages worth of content and avoid excessive detail.
        - All sections must be present in the JSON, even if empty.
        """
    else:
        prompt = f"""
        You are an expert career assistant with access to my personal portfolio and web search capabilities. My details are:
        Info: {json.dumps(data["profile"], indent=2)}
        Profile: {data['profile']['summary']}
        Experiences: {json.dumps(data['experiences'], indent=2)}
        Projects: {json.dumps(data['projects'], indent=2)}
        Skills: {', '.join(data['skills'])}
        Achievements: {json.dumps(data['achievements'], indent=2)}

        Job Description: {job_description}

        Task: Generate a {document_type} tailored to the job description. Follow these steps:
        1. Analyze the job description to identify key skills, technologies, and responsibilities.
        2. Match my skills, experiences, projects, and achievements to the job requirements.
        3. Perform a web search to gather additional context about the company or role.
        4. Structure the {document_type} to highlight relevant qualifications and achievements.
        5. Use a professional tone and ensure the content is concise and impactful.
        6. Keep it short length and use simple words and sentences.
        7. It must sound like a human and should not sound even a bit like an AI generated.
        8. Figure out the comapny's name from the job description and use it in the {document_type} wherever applicable.
        9. Use the web search results to get more details about the company to use them in the {document_type}.
        10. The length of the {document_type} should be 2-3 paragraphs with minimal and impressive content.
        11. Dont keep it generic. Write it according to me, the role and the company.
        12. Make it sound naturally human written.
        13. For a requirement in the job description which is not present in my skills or experiences mention find and mention relevant ones from my skill and say that I am a fast learner with a great grasping power and I can learn it easily.
        14. Mention that I was the co-founder of a startup and maintained it for 4 years while completing my bachelors and masters and that managerial and leadership skills will add a great value.

        Provide the {document_type} below:
        """

    response = agent.run_sync(prompt)
    return response


# Streamlit UI
def main():
    st.title("Personalized Resume & Cover Letter Generator")

    if not os.path.exists(DATA_FILE):
        st.info("Saving json file with your data...")
        save_personal_data()

    try:
        # Add dropdown for model selection
        enabled_models = [
            name for name, config in MODEL_CONFIG.items() if config["enabled"]
        ]
        if not enabled_models:
            st.error("No models enabled in MODEL_CONFIG")
            return
        if "selected_model" not in st.session_state:
            st.session_state["selected_model"] = enabled_models[0]
        selected_model = st.selectbox(
            "Select Model",
            enabled_models,
            index=enabled_models.index(st.session_state["selected_model"]),
        )
        st.session_state["selected_model"] = selected_model
        agent = LLMModel(MODEL_CONFIG[selected_model])
    except Exception as e:
        st.error(str(e))
        return

    job_description = st.text_area("Paste the Job Description", height=200)
    document_type = st.selectbox(
        "Select Document Type", ["Resume", "Cover Letter", "Email"]
    )

    if "generated_document" not in st.session_state:
        st.session_state["generated_document"] = None
    if st.button("Generate Document"):
        if job_description:
            with st.spinner(f"Generating {document_type}..."):
                try:
                    document = generate_document(job_description, document_type, agent)
                    st.session_state["generated_document"] = document
                except Exception as e:
                    st.error(f"Error generating document: {str(e)}")
                    st.session_state["generated_document"] = None
        else:
            st.error("Please provide a job description.")
            st.session_state["generated_document"] = None

    document = st.session_state["generated_document"]
    if document:
        st.subheader(f"Generated {document_type}")
        if document_type == "Resume":
            try:
                # Preprocess: Remove code block markers, lines starting with //, and trailing commas
                def clean_json_string(s):
                    s = re.sub(r"^\s*```[a-zA-Z]*", "", s, flags=re.MULTILINE)
                    s = re.sub(r"^\s*```", "", s, flags=re.MULTILINE)
                    s = re.sub(r"^\s*//.*$", "", s, flags=re.MULTILINE)
                    s = re.sub(r",(\s*[}}\]])", r"\1", s)
                    open_braces = s.count("{")
                    close_braces = s.count("}")
                    if open_braces > close_braces:
                        s += "}" * (open_braces - close_braces)
                    open_brackets = s.count("[")
                    close_brackets = s.count("]")
                    if open_brackets > close_brackets:
                        s += "]" * (open_brackets - close_brackets)
                    return s.strip()

                cleaned_doc = clean_json_string(document)
                resume_json = json.loads(cleaned_doc)
                # Set global role_name and company_name from model output if present
                role_name = resume_json.get("role_name", "")
                company_name = resume_json.get("company_name", "")
                # Display role_name and company_name as editable text boxes
                role_name_box = st.text_input(
                    "Role Name (extracted)", role_name, key="role_name_box"
                )
                company_name_box = st.text_input(
                    "Company Name (extracted)", company_name, key="company_name_box"
                )
                # Editable fields
                work_experience = []
                for i, exp in enumerate(resume_json.get("work_experience", [])):
                    with st.expander(
                        f"Experience {i+1}: {exp.get('role', '')} at {exp.get('company', '')}"
                    ):
                        role = st.text_input(
                            "Role", exp.get("role", ""), key=f"role_{i}"
                        )
                        company = st.text_input(
                            "Company", exp.get("company", ""), key=f"company_{i}"
                        )
                        duration = st.text_input(
                            "Duration", exp.get("duration", ""), key=f"duration_{i}"
                        )
                        bullets = st.text_area(
                            "Bullets (one per line)",
                            "\n".join(exp.get("bullets", [])),
                            height=120,
                            key=f"bullets_{i}",
                        )
                        work_experience.append(
                            {
                                "role": role,
                                "company": company,
                                "duration": duration,
                                "bullets": [
                                    b for b in bullets.split("\n") if b.strip()
                                ],
                            }
                        )
                st.markdown("**Projects**")
                projects = []
                for i, proj in enumerate(resume_json.get("projects", [])):
                    with st.expander(f"Project {i+1}: {proj.get('name', '')}"):
                        name = st.text_input(
                            "Name", proj.get("name", ""), key=f"proj_name_{i}"
                        )
                        description = st.text_area(
                            "Description",
                            proj.get("description", ""),
                            height=80,
                            key=f"proj_desc_{i}",
                        )
                        technologies = st.text_input(
                            "Technologies (comma separated)",
                            ", ".join(proj.get("technologies", [])),
                            key=f"proj_tech_{i}",
                        )
                        projects.append(
                            {
                                "name": name,
                                "description": description,
                                "technologies": [
                                    t.strip()
                                    for t in technologies.split(",")
                                    if t.strip()
                                ],
                            }
                        )
                st.markdown("**Skills**")
                skills = {}
                for cat, skill_list in resume_json.get("skills", {}).items():
                    val = st.text_input(
                        f"{cat} Skills (comma separated)",
                        ", ".join(skill_list),
                        key=f"skills_{cat}",
                    )
                    skills[cat] = [s.strip() for s in val.split(",") if s.strip()]
                # Download updated JSON and replace placeholders in Google Doc
                updated_json = {
                    "work_experience": work_experience,
                    "projects": projects,
                    "skills": skills,
                }
                file_name = f"{document_type.lower()}_{uuid.uuid4()}.json"
                # Google Docs integration for Resume
                st.markdown("---")
                st.markdown("### Google Docs Resume Export")
                doc_id_input = st.text_input(
                    "Google Doc Template ID (for Resume export)",
                    value="17jbwEwv7GknVg9Q1JnYnpli8aYr9GTOblGZgSbt_jcA",
                    key="resume_doc_id",
                )
                font_family = st.text_input(
                    "Font Family", "Arial", key="resume_font_family"
                )
                font_size = st.number_input(
                    "Font Size (pt)",
                    min_value=8,
                    max_value=48,
                    value=11,
                    key="resume_font_size",
                )
                bold = st.checkbox("Bold", value=False, key="resume_bold")
                # Define placeholder_map here
                placeholder_map = {
                    "SKILLS": "\n".join(
                        [f"{cat}: {', '.join(skills[cat])}" for cat in skills]
                    ),
                    "PROJECTS": "\n".join(
                        [
                            f"{proj.get('name', '')}: {proj.get('description', '')[:-1]} using {', '.join(proj.get('technologies', []))}"
                            for proj in projects
                        ]
                    ),
                    "EXPERIENCE1": (
                        "\n".join(work_experience[0]["bullets"])
                        if len(work_experience) > 0
                        else ""
                    ),
                    "EXPERIENCE2": (
                        "\n".join(work_experience[1]["bullets"])
                        if len(work_experience) > 1
                        else ""
                    ),
                    "EXPERIENCE3": (
                        "\n".join(work_experience[2]["bullets"]).strip()
                        if len(work_experience) > 2
                        else ""
                    ),
                }
                if st.button("Generate PDF", key="resume_download_btn"):
                    if doc_id_input:
                        try:
                            pdf_bytes = generate_and_download_resume_pdf_via_duplicate(
                                doc_id_input,  # This is the template doc id
                                placeholder_map,
                                font_family,
                                font_size,
                                bold,
                            )
                            st.success(
                                "Resume PDF generated from a duplicate of your template! Download will start below."
                            )
                            st.download_button(
                                label="Download Resume PDF",
                                data=pdf_bytes,
                                file_name=f"Sumanth Bejugam - {role_name} - {company_name}.pdf",
                                mime="application/pdf",
                            )
                            # Reset session state for next resume
                            for k in list(st.session_state.keys()):
                                if (
                                    k.startswith("role_")
                                    or k.startswith("company_")
                                    or k.startswith("duration_")
                                    or k.startswith("bullets_")
                                    or k.startswith("proj_")
                                    or k.startswith("skills_")
                                    or k.startswith("ach_")
                                ):
                                    del st.session_state[k]
                        except Exception as e:
                            st.error(f"Failed to generate/download PDF: {str(e)}")
                    else:
                        st.error("Please provide a Google Doc Template ID.")
            except Exception as e:
                st.error(f"Could not parse or display JSON: {str(e)}")
                st.markdown("**Raw Output:**")
                st.code(document, language="json")
                file_name = f"{document_type.lower()}_{uuid.uuid4()}.txt"
                st.download_button(
                    label="Download Document",
                    data=document,
                    file_name=file_name,
                    mime="text/plain",
                )
        else:
            st.write(document)
            file_name = f"{document_type.lower()}_{uuid.uuid4()}.txt"
            st.download_button(
                label="Download Document",
                data=document,
                file_name=file_name,
                mime="text/plain",
            )


def generate_and_download_resume_pdf_via_duplicate(
    template_doc_id, placeholder_map, font_family="Arial", font_size=11, bold=False
):
    """
    1. Duplicates the Google Doc (template_doc_id) using Drive API.
    2. Replaces placeholders in the duplicate using Docs API.
    3. Downloads the duplicate as PDF.
    4. Deletes the duplicate.
    Returns: (pdf_bytes, duplicate_doc_id)
    """
    import io
    import time
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    SCOPES = [
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = get_google_creds(SCOPES)
    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    # 1. Duplicate the doc
    copied_file = (
        drive_service.files()
        .copy(
            fileId=template_doc_id, body={"name": f"Resume Export {int(time.time())}"}
        )
        .execute()
    )
    duplicate_doc_id = copied_file["id"]
    # 2. Replace placeholders in the duplicate
    requests = []
    for ph, val in placeholder_map.items():
        requests.append(
            {
                "replaceAllText": {
                    "containsText": {"text": f"{{{{{ph}}}}}", "matchCase": True},
                    "replaceText": val,
                }
            }
        )
    if requests:
        docs_service.documents().batchUpdate(
            documentId=duplicate_doc_id, body={"requests": requests}
        ).execute()
    # 3. Download as PDF
    request = drive_service.files().export_media(
        fileId=duplicate_doc_id, mimeType="application/pdf"
    )
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    pdf_bytes = fh.read()
    # 4. Delete the duplicate
    drive_service.files().delete(fileId=duplicate_doc_id).execute()
    return pdf_bytes


if __name__ == "__main__":
    main()
