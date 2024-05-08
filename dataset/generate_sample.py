
import pickle
import datetime
from dataset.utils import NewsSource, News

headline = "Manufacturing for tomorrow: Microsoft announces new industrial AI innovations from the cloud to the factory floor"
date = datetime.datetime(2024, 4, 17)
text = """After years of uncertainty from supply chain disruption and increased customer expectations, to changes in consumer demand and workforce shortages, manufacturing remains one of the most resilient and complex industries. Today, we are witnessing the manufacturing industry enter a transformative era, fueled by AI and new AI-powered industrial solutions. This AI-driven shift is prompting many organizations to fundamentally alter their business models and re-evaluate how to address industry-wide challenges like data siloes from disparate data estates and legacy products, supply chain visibility issues, labor shortages, and the need for upskilling employees, among others.  AI is more than just an automation tool, it’s a catalyst for innovation, efficiency and sustainability. AI innovation creates an opportunity to help manufacturers enhance time-to-value, bolster operations resilience, optimize factory and production costs and produce repeatable outcomes.

Ahead of Hannover Messe, one of the world’s largest manufacturing innovation events, Microsoft is announcing new AI and data solutions for manufacturers to help unlock innovation, enable intelligent factories, optimize operations and enhance employee productivity. The manufacturing industry has been incredibly resilient over the last decade and the infusion of new AI solutions signifies a critical transformation in this vital industry.
Unlock innovation and fuel the next generation of intelligent factories with data and AI

Manufacturing is one of the most data-intensive industries, generating an average of 1.9 petabytes worldwide every year, according to McKinsey Global Institute. And most of this data goes unused, leaving many valuable insights untapped. According to Gartner® Research, “Generative AI will transform the manufacturing industry to a level previously not available, by providing new insights and recommendations based on data and actionable information.”[1] In this era of AI, the importance of data continues to grow as organizations realize they are only scratching the surface of what’s possible.

To help customers leverage their data and insights, today, we are announcing the private preview of manufacturing data solutions in Microsoft Fabric, and copilot template for factory operations on Azure AI. These solutions help manufacturers unify their operational technology and information technology (IT) data estate and accelerate and scale data transformation for AI in Fabric, our end-to-end analytics SaaS-based platform. Copilot template for factory operations helps manufacturers to create their own copilots for front-line workers utilizing the unified data. Front-line employees can use natural language to query the data for knowledge discovery, training, issue resolution, asset maintenance and more. For example, if a factory plant manager wants to understand why a machine is breaking, they can query the copilot to get insights and resolve the issue in just days, instead of weeks.

As part of our private preview, Intertape Polymer Group (IPG) uses Sight Machine’s Manufacturing Data Platform to continuously transform data generated by its factory equipment into a robust data foundation for analyzing and modeling its machines, production processes and finished products. IPG is now using Sight Machine’s Factory CoPilot, a generative AI platform with an intuitive natural language chat interface, powered by the Microsoft Cloud for Manufacturing and the copilot template for factory operations on Azure AI. This tool facilitates the team’s ability to rapidly gather insights and direct work on production lines which previously operated like black boxes. Instead of working through manual spreadsheets and inaccessible data, all teammates including production, engineering, procurement and finance have better information to drive decisions on products and processes throughout the plant improving yield and reducing inventory levels.

Also in private preview, Bridgestone is partnering with Avanade to confront production challenges head-on, focusing on critical issues related to production disruptions and scheduling inefficiencies, like yield loss, which can escalate into quality issues. As a private preview customer collaborating with Avanade, Bridgestone aims to harness the power of manufacturing data solutions in Fabric and the copilot template for factory operations. Their goal is to implement a natural language query system that enables front-line workers, with different levels of experience, with insights that lead to faster issue resolution. The team is excited to establish a centralized system that efficiently gathers and presents critical information from various sources and facilitates informed decision-making and enhances  operational agility across Bridgestone’s production ecosystem.
Creating more resilient operations and supply chains

A robust data strategy must span from cloud to the shop floor to enable the level of scale and integration that will help manufacturers accelerate industrial transformation across all operations. However, gathering OT data and integrating the data into multiple solutions is not an easy task for manufacturers. Production is complex, and their sensors, machines and systems are highly varied. Each site is unique and ensuring the right data is being shared with the right person at the right time is onerous and costly. Unfortunately, these scale and integration hurdles also block the enterprise from scaling AI solutions across every shop floor or gaining global visibility across all their sites.

With this in mind, Microsoft recently launched the adaptive cloud approach, including Azure IoT Operations. Our adaptive cloud is a framework to modernize edge infrastructure across operations, like factories, to take advantage of a modern, composable and connected architecture for your applications. Our adaptive cloud approach creates the level of scale needed to repeat AI solutions across production lines and sites. Putting the adaptive cloud approach into practice, Azure IoT Operations leverages open standards and works with Microsoft Fabric to create a common data foundation for IT and OT collaboration. To find out more about our adaptive cloud approach and Azure IoT Operations, visit our Azure Blog.

Looking to increase global operational efficiency, Microsoft’s customer Electrolux Group, developed a single platform to build, deploy and manage several key manufacturing use cases. Their platform’s goal is to capture all manufacturing data, contextualize it and make it available for real time decision-making across all levels of the organization within a scalable infrastructure. To enable this, Electrolux Group is adopting a full stack solution from Microsoft that leverages the adaptive cloud approach, including Azure IoT Operations. Using this approach, Electrolux Group is looking to reduce overhead from multiple vendors, a consistent and simple way to deploy and manage multiple use cases at a site, and then the ability to scale those solutions to multiple sites with simple and consistent fleet management.

Supply chain disruption is not new; however, its complexity and the rate of change are outpacing organizations’ ability to address issues. Manufacturers are under pressure to prevent and minimize disruptions, and as a result, almost 90% of supply chain professionals plan to invest in ways to make their supply chains more resilient. To support our customers, we’re announcing the upcoming preview of a traceability add-in for Dynamics 365 Supply Chain Management that will allow businesses to increase visibility into their product genealogy through the different steps of the supply chain. Traceability will also help businesses track events and attributes throughout supply chain processes and will provide an interface to query and analyze data.
Empowering front-line workers with AI tools to improve productivity,and job satisfaction

To enable intelligent factory operations, an empowered and connected workforce is key. According to the latest Work Trend Index, 63% of front-line workers do repetitive or menial tasks that take time away from more meaningful work. Additionally, 80% of front-line workers think AI will augment their ability to find the right information and the answers they need. From the office to the factory floor to the field, we are building solutions to address the unique challenges manufacturers face — by helping streamline front-line operations, enhance communication and collaboration, improve employee experience and strengthen security across shared devices.

Today we’re introducing new capabilities for Copilot in Dynamics 365 Field Service that help service managers and technicians efficiently find information, resolve issues while keeping customers updated at every step, and help summarize their work. Generally available, field service managers can interact with Copilot to find pertinent information about work orders using natural language in their flow of work in the Dynamics 365 Field Service web app. Additionally, available in public preview, front-line workers can configure and customize the fields Copilot uses to generate summaries within Dynamics 365 Field Service.

To further streamline collaboration among field service managers, technicians, and remote experts, Dynamics 365 Field Service users with the Field Service app in Teams can now share links to work orders that automatically expand to provide key details. This capability is generally available starting today. Should technicians need additional assistance from remote experts to resolve issues, they can simply access Dynamics 365 Remote Assist capabilities in the flow of work in Microsoft Teams with anchored spatial annotations even if the camera moves.
Microsoft ecosystem and partnerships in the era of AI

These new industry innovations in data and AI are strengthened through the Microsoft Cloud for Manufacturing, which enables organizations to accelerate their data and AI journey by augmenting the Microsoft Cloud with industry-relevant data solutions, application templates and AI services. The Microsoft Cloud for Manufacturing brings the best of Microsoft and our partners to jointly accelerate the digital transformation in manufacturing.

Microsoft is a trusted co-innovation partner committed to working with enterprises to unlock the true potential of AI solutions and transform the industry.​ Our offerings can also be customized by an unmatched global ecosystem of trusted partners. This year, we’re proud to have the following valued partners demonstrate at our Hannover Messe booth: Accenture, Annata, Ansys, Avanade, AVEVA, Blue Yonder, Bosch, CapGemini, Cognite, Connected Cars DK, DSA, HERE Technologies, Hexagon, Netstar, NVIDIA, o9 Solutions, PTC, Rockwell Automation, SAP, Syntax, Sight Machine, Siemens, SymphonyAI, Tata Consultancy Services (TCS), Threedy, ToolsGroup and Tulip Interfaces.

We look forward to seeing you at the Microsoft Booth in Hall 17 Stand G06, where you can join guided tours, and speak with manufacturing and industrial experts from around the world."""

sample = News(headline, date, text, NewsSource.MICROSOFT)
with open('sample.dat', 'wb') as f:
    pickle.dump(sample, f)