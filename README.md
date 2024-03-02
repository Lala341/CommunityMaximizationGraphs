
\scarinfo{
    title={Greedy/Heuristic vs. ML for Community Detection and Influence Maximization},
    shorttitle={Delivery Massive Graph Management and Analytics},
    author={Ogbuchi Chidiebere, Forero Laura},
    shortauthor={Ogbuchi, C. and Forero, L.},
    authorline={Ogbuchi Chidiebere\textsuperscript{1} and Forero Laura\textsuperscript{2}},
    affiliation={
        \item CentraleSupélec
    },
}

\scarheaders{}

\setcounter{page}{1}

% STEP 4: Supply your own bibliography file
% Export all of your references in BibLaTeX (or BibTeX) format and save them in
% the "bibliography.bib" file.
%\addbibresource{bibliography.bib}

\begin{document}

\maketitle


% STEP 6: Write the main content
% Consult the text below for some more guidance on how the paper should be
% written. It also includes some examples on how to use citations, figures and
% tables in LaTeX. Happy writing!

\section{Datasets}


The goal of this project is to examine and contrast greedy/heuristic methods with ML methods for addressing community detection and influence maximization problems using real data from SNAP datasets. Specifically, two social networks, Facebook and English Twitch, were employed for analysis.

\subsection{Facebook}
The Facebook dataset consists of anonymized 'circles' or 'friends lists' collected from survey participants via a Facebook app. It includes node features, circles, and ego networks. The data has been anonymized by replacing Facebook-internal IDs and obscuring feature interpretations. Despite this, users' shared affiliations can be identified without revealing individual details. The dataset contains 4039 nodes and 88234 edges, with an average clustering coefficient of 0.6055 and a diameter of 8.

\begin{table}[htbp]
    \centering
    \caption{Dataset Statistics}
    \begin{tabular}{@{}ll@{}}
    \toprule
    \textbf{Statistic} & \textbf{Value} \\ \midrule
    Nodes & 4039 \\
    Edges & 88234 \\
    Nodes in largest WCC & 4039 (1.000) \\
    Edges in largest WCC & 88234 (1.000) \\
    Nodes in largest SCC & 4039 (1.000) \\
    Edges in largest SCC & 88234 (1.000) \\
    Average clustering coefficient & 0.6055 \\
    Number of triangles & 1612010 \\
    Fraction of closed triangles & 0.2647 \\
    Diameter (longest shortest path) & 8 \\
    90-percentile effective diameter & 4.7 \\ \bottomrule
    \end{tabular}
\end{table}




\subsection{Twitch}



The Twitch dataset comprises user-user networks of gamers who stream in English. Nodes represent individual users, while links denote mutual friendships between them. Vertex features are derived from games played, liked, location, and streaming habits. These networks were collected in May 2018 and are used for node classification and transfer learning tasks. The supervised task involves binary node classification to predict whether a streamer uses explicit language. The dataset contains 7,126 nodes and 35,324 edges, with a density of 0.002 and transitivity of 0.042. Additionally, tasks such as transfer learning, link prediction, community detection, and network visualization are possible with this dataset.

\begin{table}[htbp]
    \centering
    \caption{Dataset Statistics}
    \begin{tabular}{@{}ll@{}}
    \toprule
    \textbf{Statistic} & \textbf{EN} \\ \midrule
    Nodes & 7,126 \\
    Edges & 35,324 \\
    Density & 0.002 \\
    Transitivity & 0.042 \\ \bottomrule
    \end{tabular}
\end{table}


\section{Community Detection}
Community detection in complex networks has become a fundamental aspect of network analysis, revealing hidden structures and patterns within intricate systems. Networks, such as social networks, biological networks, and technological networks, often exhibit modular organization where nodes form tightly-knit groups or communities. Understanding these communities is crucial for various applications, ranging from targeted marketing strategies to identifying functional modules in biological systems. In this report, we delve into the significance of community detection and explore three widely-used methods: Clique Percolation Method, Louvain Method, and Girvan-Newman Method.\\


\subsection{Methods of Community Detection}
In this delivery we explore Traditional greedy/heuristic and ML approaches\\
\subsubsection{Methods of Community Detection}
\textbf{Clique Percolation Method (palla):}
The Clique Percolation Method (CPM)
