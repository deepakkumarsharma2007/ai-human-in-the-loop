import json
from logging import Logger
import logging
import os

from pydantic import BaseModel, Field
from typing import Any, Optional
from langchain_core.tools import BaseTool

# from core.agent import create_model, create_react_agent
# from core.auth_middleware import audit_info_decorator
# from genai_core.log_exceptions_middleware import log_exceptions_decorator
from urllib.parse import quote

from genai_core.logs.agent_logging import DKSAgentLogger
from genai_core.rag.embedding.rag_vector_search import find_relevant_chunks_from_mongodb_vector_store

class MongoDBRAGSearchAdapterSchema(BaseModel):
    query: str = Field(
        ...,
        description="Users query in natural language. The query should be related to document search in MongoDB databases.",
    ),
    # auditcontext: Optional[dict[str, Any]] = Field(
    #     default_factory=dict, description="auditcontext of the current request"
    # )


class MongoDBRAGSearchToolAdapter(BaseTool):
    """
    MongoDBRAGSearchToolAdapter is an agent adapter tool designed to handle natural language queries related to document search in MongoDB Databases.
        - It receives a natural language query and an audit context.
    """

    name: str = "MongoDB_RAG_Document_Search_Query_Tool"
    description: str = """This tool is designed to handle user query to search documents in MongoDB databases.
    Contents it can use Microsoft Azure Services like Azure Service Bus, Azure Functions, any content related to 
    Table of Contents

Section 1: Solution and Infrastructure

Getting Started as an Azure Architect
Technical requirements
Getting to know architectural duties
Enterprise architects
Domain architects
Solution architects
Data architects
Technical architects
Security architects
Infrastructure architects
Application architects
Azure architects
Architects versus engineers
Getting started with the essential cloud vocabulary
Cloud service models map
IaaS (Infrastructure as a Service)
PaaS (Platform as a Service)
FaaS (Function as a Service)
CaaS (Containers as a Service)
DBaaS (Database as a Service)
XaaS or *aaS (Anything as a Service)
Introducing Azure architecture maps
How to read a map
Understanding the key factors of a successful cloud journey
Defining the vision with the right stakeholders
Defining the strategy with the right stakeholders
Starting implementation with the right stakeholders

Solution Architecture
Technical requirements
The solution architecture map
Zooming in on the different workload types
Understanding systems of engagement
Understanding systems of record
Understanding systems of insight
Understanding systems of interaction (IPaaS)
Looking at cross-cutting concerns and non-functional requirements
Looking at cross-cutting concerns and the cloud journey
Zooming in on containerization
Solution architecture use case
Looking at a business scenario
Using keywords
Using the solution architecture map against the requirements
Building the target reference architecture
Code view of our workflow-based reference architecture
Looking at the code in action
Understanding the gaps in our reference architecture Summary

Infrastructure Design
Technical requirements
The Azure infrastructure
architecture map
Zooming in on networking
The most common architecture
Data center connectivity options
Zoning
Routing and firewalling
Zooming in on monitoring
Zooming in on high availability
and disaster recovery
Zooming in on backup and
restore
Zooming in on HPC
AKS infrastructure
Exploring networking options with AKS
Exploring deployment options with AKS
Monitoring AKS
Exploring AKS storage options
Scaling AKS
Exploring miscellaneous aspects
AKS and service meshes for
microservices versus Azure native
services
AKS reference architecture for
microservices – cluster boundaries
AKS reference architecture for
microservices – cluster internals

Infrastructure Deployment
Technical requirements
Introducing Continuous
Integration and Continuous
Deployment (CI/CD)
Introducing the CI/CD process
Introducing the IaC CI/CD process
The Azure deployment map 124
Getting started with the Azure
CLI, PowerShell, and Azure
Cloud Shell 127
Playing with the Azure CLI from within
Azure Cloud Shell 127
Using PowerShell from within Azure
Cloud Shell 132
Combining PowerShell and the Azure
CLI from within Azure Cloud Shell 134
Understanding the one that
rules them all 135
Diving into ARM templates 137
Getting started with ARM 137
Understanding the ARM template
deployment methods 137
Understanding the ARM template
deployment scopes 138
Understanding the ARM template
deployment modes 142
Understanding the anatomy of an ARM
template 144
Building a concrete example using
linked templates 147
Getting started with Azure
Bicep 159
Getting started with Terraform 162
Zooming in on a reference
architecture with Azure DevOps 168
Using a simple approach to an IaC
factory 169
Using an advanced approach to an IaC
factory 172

Application Architecture 179
Technical requirements 180
Understanding cloud and
cloud‑native development 181
Exploring the Azure Application
Architecture Map 183
Zooming in on data 185
Zooming in on cloud design patterns 186
Dealing with cloud-native patterns 193
Understanding the COMMODITIES
top‑level group 201
iv Table of Contents
Exploring EDAs 204
Inspecting the Azure Service Bus
configuration 210
Adding the other components to the
mix 214
Developing microservices 216
Using Dapr for microservices 217
Understanding Dapr components 219
Getting started with Dapr SDKs 220
Looking at our scenario 222
Developing our solution 223
Testing our solution 229
Combining Dapr and the API gateway
of Azure APIM 232

Data Architecture 239
Technical requirements 240
Looking at the data
architecture map 240
Analyzing traditional data
practices 242
Introducing the OLAP and OLTP
practices 243
Introducing the ETL practice 243
Introducing the RDBMS practice 244
Delving into modern data
services and practices 245
Introducing the ELT practice 246
Exploring NoSQL services 246
Learning about object stores 248
Diving into big data services 249
Ingesting big data 250
Exploring big data analytics 251
Azure-integrated open source big data
solutions 253
Introducing AI solutions 253
Understanding machine learning and
deep learning 254
Integrating AI solutions 256
Dealing with other data
concerns 257
Introducing Azure Cognitive Search 257
Sharing data with partners and
customers (B2B) 258
Migrating data 258
Governing data 259
Getting our hands dirty with a
near real-time data streaming
use case 259
Setting up the Power BI workspace 260
Setting up the Azure Event Hubs
instance 260
Setting up Stream Analytics (SA) 261
Testing the code 263

Security Architecture 267
Technical requirements 268
Introducing cloud-native
security 268
Reviewing the security
architecture map 270
Exploring the recurrent services
security features 272
Exploring the recurrent data services
security features 280
Zooming in on encryption 282
Managing your security posture 286
Zooming in on identity 290
Delving into the most recurrent
Azure security topics 294
Exploring Azure managed identities in
depth 294
Demystifying SAS 297
Understanding APL and its impact on
network flows 298
Understanding Azure resource firewalls 301


What this book covers
Chapter 1, Getting Started as an Azure Architect, starts by sharing a view of the different
architecture disciplines. We define the roles and responsibilities of the various architects
(enterprise, solution, infrastructure, data, and security). The rationale of going through
these definitions lies in the fact that, from our experience, we have noticed some knowledge
gaps in what the different stakeholders are doing. This often leads to turf wars, which can
be avoided simply by understanding the broader picture. We then introduce our maps, and
we help you understand how to properly conduct a cloud strategy and what the key aspects
are that will make your cloud journey successful. In a nutshell, we give you a glimpse into
what it feels like to be an Azure architect who has to deal with all these different disciplines,
and who sometimes must report to top management on strategic aspects.
Chapter 2, Solution Architecture, covers key aspects to consider when building a cloud
solution. A solution architect is responsible for the end-to-end aspects of a solution, from
its development to its monitoring. A solution architect knows what Agile methodologies
are, as well as what ITIL, TOGAF, and COBIT are. They are the cornerstone of a solution,
its main pillar. The primary role of a solution architect is to assemble all the building
blocks to make a consistent and coherent design, as well as to talk to various stakeholders.
Their stakeholders are other, more specialized architects, developers, and IT engineers,
as well as enterprise architects and management. This chapter remains high-level from
a technical perspective because we will still envision Azure as a whole. We share the
solution architecture map, which encompasses many Azure services, and we explore
multiple dimensions around the non-functional requirements. We also zoom in on
Azure's container platform offering, which has been booming and expanding greatly over
the last few years. Lastly, we will walk you through a concrete use case and a glimpse into
what comes next, including a deeper dive into the technical and technological aspects.
Chapter 3, Infrastructure Design, delves deeper into technical matters. We will review the
typical infrastructure topologies and we will zoom into infrastructure-specific concerns
such as networking, monitoring, backup and restore, high availability, and disaster
recovery (for which we'll see a sample use case). Because containerization has become
mainstream, we will also dive into Azure Kubernetes Services (AKS) and unveil a
dedicated AKS architecture map. You will learn that AKS is not really a service like the
others, and we will walk you through a reference architecture to host a service mesh (for
microservices) in AKS.
Chapter 4, Infrastructure Deployment, is almost entirely hands-on! You will learn about
the different Infrastructure as Code (IaC) tools and frameworks. You will provision
some Azure services using Azure Resource Manager templates, Bicep, and Terraform.
Nevertheless, we won't forget our architecture glasses, so we will also look at the machinery
of a Continuous Integration and Continuous Delivery/Deployment (CI/CD) factory.

Chapter 5, Application Architecture, looks at what the development architecture would
look like for building an app on the Microsoft cloud. You may ask 10 different people
what cloud-native means, and you might receive 10 different answers. So, we will start by
explaining what we mean when we refer to the cloud and cloud-native solutions. Next, we
will review some modern design patterns, such as CQRS, Event Sourcing, and so on. In
the process, we will map them to the Azure services to help you identify how to bundle
the services together in order to build solutions based on these patterns. Lastly, we will
go through a microservices use case, using Dapr (Distributed Application Runtime),
which is a very recent and promising framework for developing distributed applications.
Throughout this chapter, our motto will be to not reinvent the wheel. Instead, leverage the
ecosystem to design and build your solutions.
Chapter 6, Data Architecture, explores how data is processed and stored. Data is the new
gold, and Azure contains many gold mines! In this chapter, we will consider traditional
and modern data practices in opposition to each other, and see how to use both in
Azure. We will also explore big data and artificial intelligence and analytics. At last, our
hands-on use case is based on a data-streaming scenario. We are going to build a real-time
dashboard, which consolidates aggregates of metrics from a fake speed detector (which
we have developed for you). A separate real-time tile will show all the vehicles that should
receive a fine (for breaking the law).
Chapter 7, Security Architecture, emphasizes and explains the importance of security in
the cloud. Security is everywhere, and it's even more important with the cloud. This tends
to awaken age-old fears and trepidations. This topic certainly deserves an entire book,
so (to avoid writing a second book) we decided to be very pragmatic and to focus on the
essential parts only. We start by giving you a glimpse into cloud-native security, to see
beyond the technology and what the required mindset is. We will then explain why there
is a paradigm shift in identity with the public cloud, by simply … proving it! Lastly, we
will focus on the most recurrent security services and topics in Azure, which you must
absolutely master as an Azure architect. Throughout the chapter, our motto will be to not
simply stack network layers. Instead, think further and modernize your security practices.
Chapter 8, Summary and Industry Scenarios, revisits the topics covered in the book and
consolidates our key ideas from each previous chapter. In other words, we'll identify what
the most important aspects to remember are. In addition, we'll look at several key industry
verticals through the lens of the previous chapters, to guide you through some existing
architectures that you can continue exploring after you complete the book. We'll finish
with some notes on the unique key values of this book, and a brief summary.



    """
    
    args_schema: type[MongoDBRAGSearchAdapterSchema] = MongoDBRAGSearchAdapterSchema
    logger: Logger = DKSAgentLogger.get_logger(__name__)

    # @audit_info_decorator()
    # @log_exceptions_decorator()
    async def _arun(
        self,
        query: str,
        # ctx: Context,
        auditcontext: Optional[dict[str, Any]] = None,
    ) -> str:
        query = quote(query, safe="")
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
    
        result = await self.find_documents_rag_search(query)

        return result if isinstance(result, str) else json.dumps(result)

    def _run(self) -> Any:
            raise NotImplementedError("This tool is async only. Use _arun method.")
    
    async def find_documents_rag_search(self, user_query: str)-> list[str]:
        """
        Does semantic search in MongoDB RAG based index.
        """
        try:
            # Log the user query
            self.logger.debug(f"Received natural language query: {user_query}")

            results = await find_relevant_chunks_from_mongodb_vector_store(user_query)
        
            return [result.page_content for result in results]
        
        except Exception as e:
            self.logger.error(f"Error processing natural language query {user_query} \n\n exception: {e}")
            raise

def mongodb_document_search_agent_adapter() -> MongoDBRAGSearchToolAdapter:
    """
    Returns MongoDB Natural Language Query agent adapter tool instance.
    """
    mongodburl = os.getenv("MONGODB_URI")
    if mongodburl is None:
        raise ValueError(
            "Environment variable 'MONGODB_URI' is not set."
        )
    mongodb_database_name = os.getenv("MONGODB_NATURAL_LANGUAGE_DATABASE_NAME", "")
    if not mongodb_database_name:
        raise ValueError(
            "Environment variable 'MONGODB_NATURAL_LANGUAGE_DATABASE_NAME' is not set or empty."
        )
    mongodb_document_search_agent_adapter = MongoDBRAGSearchToolAdapter()

    return mongodb_document_search_agent_adapter