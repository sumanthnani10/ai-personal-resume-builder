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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL_CONFIG = {
    "grok": {
        "enabled": True,
        "api_key": os.getenv("XAI_API_KEY"),
        "model_name": "grok-3-beta",
        "base_url": "https://api.x.ai/v1",
        "chat_endpoint": "/chat/completions",
        "fallback_models": [
            "grok-3-fast-beta",
            "grok-3-mini-beta",
        ],
    },
    "openai": {
        "enabled": False,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "chat_endpoint": "/chat/completions",
    },
    "gemini": {
        "enabled": False,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "model_name": "gemini-1.5-pro",
    },
    "anthropic": {
        "enabled": False,
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "model_name": "claude-3-5-sonnet-latest",
    },
}
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MODEL_NAME = "distilgpt2"
DATA_FILE = "personal_data.json"
OUTPUT_DIR = "./fine_tuned_model"

# Sample personal data (unchanged)
personal_data = {
    "profile": {
        "name": "Sumanth Bejugam",
        "email": "sumanthbejugam@gmail.com",
        "phone": "5714650804",
        "linkedin": "https://linkedin.com/in/bejugamsumanth",
        "summary": "Full Stack Developer with 4+ years of experience building scalable web, mobile, and IoT applications using Flutter, React, Kotlin, Node.js, and Python. Proven track record of delivering production-ready solutions under tight deadlines, with 50,000+ app downloads, enterprise adoption, and government recognition. Founder of a tech startup and winner of multiple hackathons. Skilled in cloud automation (AWS, Azure), CI/CD, ML, and BLE-based IoT systems. Known for rapid learning, clean architecture, and cross-platform expertise. Eligible for Public Trust Clearance.",
    },
    "experiences": [
        {
            "role": "Full Stack Developer",
            "company": "Epitome Research And Innovation Inc, Herndon, VA",
            "duration": "Sep 2024 – Present",
            "description": [
                "Engineered a responsive web application using React with Next.js and Mantine, reducing page load times from 5 seconds to 3 seconds.",
                "Optimized user-facing interfaces for simulation management in a high-security environment, reducing database interaction time from 1 second to 0.6 seconds.",
                "Designed and implemented RESTful APIs with Flask, reducing development cycles from 20 weeks to 15 weeks.",
                "Implemented 3D object visualization in React using Three.js, cutting STL file rendering time from 10 seconds to 5 seconds.",
                "Designed a scalable MongoDB data storage solution, reducing query latency from 200 ms to 130 ms.",
                "Orchestrated API and website hosting on AWS EC2 instances, reducing server response times from 500 ms to 325 ms.",
                "Automated AWS infrastructure management using Boto3 for EC2, Elastic File System (EFS), and Elastic IP (EIP), reducing monthly costs from 1000 to 700.",
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
            "name": "Red And Blue Appliance Repair",
            "description": "A responsive website developed for Red And Blue Appliance Repair, an appliance repair service company. The website showcases the company's services, provides a contact form for user inquiries, and includes features like SEO optimization for better visibility. The design was created with a focus on ease of navigation and user engagement. The website has a modern tech stack, ensuring fast load times and seamless user experience. It also integrates SEO to increase search engine rankings, which contributed to over 100 leads in the first week of launch.",
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
                "SEO",
                "Digital Marketing",
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
            "technologies": ["Flutter", "BLE", "MQTT", "Mosquitto", "Arduino C", "NodeJS", "ExpressJS"],
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
}


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

# Fine-tune the model
def fine_tune_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_dataset()

    def tokenize_function(examples):
        encodings = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

# Generic model class for API-based LLMs
class LLMModel:
    def __init__(self, model_config):
        self.model_name = model_config["model_name"]
        self.api_key = model_config["api_key"]
        self.base_url = model_config["base_url"]
        self.chat_endpoint = model_config["chat_endpoint"]
        self.fallback_models = model_config.get("fallback_models", [])
        self.enabled = model_config["enabled"]

    def run_sync(self, prompt):
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
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        api_url = f"{self.base_url}{self.chat_endpoint}"
        logger.debug(f"Attempting API call to {api_url} with model {self.model_name}")

        models_to_try = [self.model_name] + self.fallback_models
        for model in models_to_try:
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
    return LLMModel(MODEL_CONFIG[model_name])

# Function to extract company and role name using regex
def extract_company_name(job_description):
    role_name = None
    company_name = None

    # Patterns to find role name
    role_patterns = [
        r"(?:Position|Role|Job Title):\s*([A-Za-z\s/]+?)(?:,|\n|$)", # Position: Software Engineer
        r"\b(Software Engineer|Developer|Analyst|Architect|Manager|Lead|Intern)\b", # Common role keywords
        r"hiring\s+a\s+([A-Za-z\s/]+)", # hiring a Software Engineer
        r"looking\s+for\s+a\s+([A-Za-z\s/]+)", # looking for a Software Engineer
    ]

    # Patterns to find company name
    company_patterns = [
        r"(?:Company|Organization):\s*([A-Z][a-zA-Z0-9&\s.,]+?)(?:,|\n|$)", # Company: Google LLC
        r"\bat\s+([A-Z][a-zA-Z0-9&\s.,]+?)(?:,|\s|\n|$)", # at Google Inc.
        r"\b([A-Z][a-zA-Z0-9&\s.,]+?)\s+(?:is hiring|seeks|Inc\.|LLC|Corp)", # Google LLC is hiring
    ]

    # Find role name
    for pattern in role_patterns:
        match = re.search(pattern, job_description, re.IGNORECASE)
        if match:
            potential_role = match.group(1).strip()
            # Avoid overly generic matches or parts of sentences
            if len(potential_role.split()) <= 4 and potential_role.lower() not in ["a", "an", "the"]:
                 role_name = potential_role
                 break # Found a plausible role

    # Find company name
    for pattern in company_patterns:
        match = re.search(pattern, job_description)
        if match:
            potential_company = match.group(1).strip().rstrip(',.')
            # Basic check to avoid matching common words or very short strings
            if len(potential_company) > 2 and (potential_company.istitle() or any(c.isupper() for c in potential_company)):
                 # Further clean up common suffixes if needed
                 potential_company = re.sub(r'\s+(Inc\.?|LLC|Corp\.?)$', '', potential_company, flags=re.IGNORECASE).strip()
                 company_name = potential_company
                 break # Found a plausible company

    if role_name and company_name:
        # Refine role name if it contains the company name (e.g., "Software Engineer at Google")
        if company_name.lower() in role_name.lower():
             role_name = role_name.lower().replace(f"at {company_name.lower()}", "").strip().title()

        print(f"Extracted Role: {role_name}, Company: {company_name}")
        return f"{role_name} at {company_name}"
    elif company_name:
         print(f"Extracted Company only: {company_name}")
         return company_name # Return company name if role not found
    elif role_name:
         print(f"Extracted Role only: {role_name}")
         # Decide if returning only role makes sense, maybe not for search query
         return None

    print("Could not extract Role or Company name reliably.")
    return None

# Web search function
def web_search(query):
    print(f"Performing web search for query: {query}")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            snippets = [result["body"] for result in results]
        return "\n".join(snippets)
    except Exception as e:
        return f"Web search failed: {str(e)}"

# Updated generate_document function
def generate_document(job_description, document_type, agent):
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    # Try to extract company name
    company_name = extract_company_name(job_description)
    if company_name:
        search_query = f"{company_name} job requirements"
    else:
        # Fallback to first two lines of job description
        lines = job_description.split("\n")
        search_query = " ".join(lines[:2]).strip()

    web_search_results = web_search(search_query)
    print(f"Web Search Query: {search_query}")
    print(f"Web Search Results: {web_search_results}")

    if document_type == "Resume":
        prompt = f"""
        Here is my personal portfolio data:
        Profile: {data['profile']['summary']}
        Experiences: {json.dumps(data['experiences'], indent=2)}
        Projects: {json.dumps(data['projects'], indent=2)}
        Skills: {', '.join(data['skills'])}
        Achievements: {json.dumps(data['achievements'], indent=2)}

        Job Description: {job_description}
        Web Search Results: {web_search_results}

        Generate a resume tailored to the job description, using the following format:

        **Sumanth Bejugam**  
        Fairfax, VA | sumanthbejugam@gmail.com | +1 (571) 465-0804  
        https://linkedin.com/in/bejugamsumanth | https://github.com/sumanthnani10 | https://sumanthbejugam.github.io/portfolio/  

        **OVERVIEW**  
        [Generate a 2-3 sentence summary highlighting my most relevant skills, experiences, and achievements based on the job description. Use insights from the web search results to align with the company’s focus or job requirements.]

        **WORK EXPERIENCE**  
        [Include all relevant work experiences from the personal data, prioritizing those most aligned with the job description. For each, provide:  
        - **Role, Company (Duration)**  
          - Bullet point 1: Key achievement or responsibility  
          - Bullet point 2: Another key point  
          - Bullet point 3: Another key point  
          - Bullet point 4: Another key point  
          - Bullet point 5: Another key point  
          - Bullet point 6: Another key point and so on, up to 8 points with atleast 4 points.
        Use the detailed descriptions in the personal data to craft concise bullet points emphasizing skills and achievements relevant to the job. Keep the count of points according to the time in that company. When describing achievements or improvements, always prefer using absolute metrics (e.g., "reduced page load times from 5s to 3s") or concrete numbers (e.g., "reduced server response time by 200 ms"). Only use percentages if no other specific metric is available.]

        **PROJECTS**  
        [Select 3-4 projects that best demonstrate skills or experiences pertinent to the job. For each, provide:  
        - **Project Name**: Brief description (1 sentence) highlighting relevance to the job and include Technologies used.
        Use the project descriptions in the personal data and enhance with relevant web search insights if applicable.]

        **ACHIEVEMENTS**  
        [List 3-4 notable achievements from the personal data that showcase capabilities related to the job.]

        **SKILLS**  
        [Select and list skills from the personal data that match or are implied by the job description, categorize them and for each category:
        - **Category**: Skill 1, Skill 2, Skill 3, .....]

        Additional Instructions:  
        - Prioritize content that directly matches the job’s key requirements (skills, technologies, responsibilities).  
        - Use web search results to tailor the overview and experience descriptions to the company’s values or additional job context.  
        - Keep content concise, professional, and impactful, limiting bullet points to 2-7 per experience.  
        - Format using markdown with bold headings and bullet points as shown.
        - Limit the resume to 1-2 pages in length and avoid excessive detail.

        Provide the resume below:
        """
    else:
        prompt = f"""
        You are an expert career assistant with access to my personal portfolio and web search capabilities. My details are:
        Info: {json.dumps(data["profile"], indent=2)}
        # Profile: {data['profile']['summary']}
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
        11. Dont keep it generic. Write it according to me.
        12. Make it sound naturally human written.

        Web Search Results: {web_search_results}

        Provide the {document_type} below:
        """

    response = agent.run_sync(prompt)
    return response

# Streamlit UI
def main():
    st.title("Personalized Resume & Cover Letter Generator")

    if not os.path.exists(OUTPUT_DIR):
        st.info("Fine-tuning model with your data...")
        save_personal_data()
        try:
            fine_tune_model()
            st.success("Model fine-tuned successfully!")
        except Exception as e:
            st.error(f"Error during fine-tuning: {str(e)}")
            return

    try:
        agent = initialize_agent()
    except Exception as e:
        st.error(str(e))
        return

    job_description = st.text_area("Paste the Job Description", height=200)
    document_type = st.selectbox(
        "Select Document Type", ["Resume", "Cover Letter", "Email"]
    )

    if st.button("Generate Document"):
        if job_description:
            with st.spinner(f"Generating {document_type}..."):
                try:
                    document = generate_document(job_description, document_type, agent)
                    st.subheader(f"Generated {document_type}")
                    st.write(document)

                    file_name = f"{document_type.lower()}_{uuid.uuid4()}.txt"
                    st.download_button(
                        label="Download Document",
                        data=document,
                        file_name=file_name,
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Error generating document: {str(e)}")
        else:
            st.error("Please provide a job description.")


if __name__ == "__main__":
    main()
