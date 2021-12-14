import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import webbrowser
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='NBA Workstation',
                   page_icon="curry.jpg",
                   layout="wide")

rad = st.sidebar.radio("Navigation", ["HOME", "TEAM STATS", "PLAYER STATS", "PLAYER COMPARISON", "PERFORMANCE INDEX", "LOVE FOR THE GAME"])

if (rad == "HOME"):
    st.title("NBA WORKSTATION")
    st.markdown("""
    **WEB APP for Analyzing NBA Players Stats**
    * **Python libraries:** Pandas, Streamlit
    * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/) and NBA_Api
    """)
    st.image("home3.jpg", width=1180)


elif (rad == "TEAM STATS"):

    st.title('TEAM STATS')

    st.sidebar.header('User Input Features')
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2022))))


    # Web scraping of NBA player stats
    @st.cache
    def load_data(year):
        url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
        html = pd.read_html(url, header=0)
        df = html[0]
        raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
        raw = raw.fillna(0)
        playerstats = raw.drop(['Rk'], axis=1)
        return playerstats


    playerstats = load_data(selected_year)

    # Sidebar - Team selection
    sorted_unique_team = sorted(playerstats.Tm.unique())
    selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

    # Sidebar - Position selection
    unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

    # Filtering data
    df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

    st.subheader('Display Player Stats of Selected Team(s)')
    st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(
        df_selected_team.shape[1]) + ' columns.')
    st.dataframe(df_selected_team)


    # Download NBA player stats data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
        return href


    st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

    # Heatmap
    if st.button('Intercorrelation Heatmap'):
        st.header('Intercorrelation Matrix Heatmap')
        df_selected_team.to_csv('output.csv', index=False)
        df = pd.read_csv('output.csv')

        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()


elif (rad == "PLAYER STATS"):
    player_dict = players.get_players()
    # Conversion Of Dictionary To Dataframe
    player_data = pd.DataFrame.from_dict(player_dict)
    player_info = st.sidebar.selectbox("Enter Player Name: ", player_data['full_name'], index= 871)
    selected_player = [player for player in player_dict if player['full_name'] == player_info][0]
    id_select = selected_player['id']

    stat_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))))

    gamelog_player = playergamelog.PlayerGameLog(player_id=id_select, season=stat_year)
    df_gamelog = gamelog_player.get_data_frames()[0]

    st.title("PLAYER STATS")
    st.subheader(player_info)
    st.write(player_data.loc[player_data['full_name'] == player_info])
    st.subheader("NBA Season " + str(stat_year) + "-" + str(stat_year + 1) + " Stats")
    st.write(df_gamelog)

    max_pts = df_gamelog.sort_values('PTS', ascending=False)
    top_five_offence = max_pts.head(5)

    max_ast = df_gamelog.sort_values('AST', ascending=False)
    top_five_playmaking = max_ast.head(5)

    max_reb = df_gamelog.sort_values('REB', ascending=False)
    top_five_defence = max_reb.head(5)

    b1 = st.button("Season Points Graph")
    if (b1):
        best_chart = alt.Chart(df_gamelog).mark_point().encode(
            x='MATCHUP', y="PTS", tooltip=['MATCHUP', 'PTS']
        )
        st.altair_chart(best_chart)
        # sns.lineplot(x="MATCHUP", y="PTS", data=df_gamelog)
        # col1.pyplot()
    b2 = st.button("Season Rebounds Graph")
    if (b2):
        best_chart = alt.Chart(df_gamelog).mark_point().encode(
            x='MATCHUP', y="PTS", tooltip=['MATCHUP', 'PTS']
        )
        st.altair_chart(best_chart)
        # sns.lineplot(x="MATCHUP", y="REB", data=df_gamelog)
        # col2.pyplot()

    c1, c2, c3 = st.beta_columns(3)
    if c1.button("Top 5 Offensive Matches"):
        best_chart = alt.Chart(top_five_offence).mark_circle().encode(
            x='MATCHUP', y="PTS", tooltip=['MATCHUP', 'PTS']
        )
        c1.altair_chart(best_chart)

    if c2.button("Top 5 Playmaking Matches"):
        best_chart = alt.Chart(top_five_playmaking).mark_circle().encode(
            x='MATCHUP', y="AST", tooltip=['MATCHUP', 'AST']
        )
        c2.altair_chart(best_chart)

    if c3.button("Top 5 Defencive Matches"):
        best_chart = alt.Chart(top_five_defence).mark_circle().encode(
            x='MATCHUP', y="REB", tooltip=['MATCHUP', 'REB']
        )
        c3.altair_chart(best_chart)

        # sns.lineplot(x = "MATCHUP", y ="PTS", data=top_five)
        # col2.pyplot()

    if st.button("Career Stats"):
        gamelog_player_all = playergamelog.PlayerGameLog(player_id=id_select, season=SeasonAll.all)
        df_gamelog_all = gamelog_player_all.get_data_frames()[0]
        st.write(df_gamelog_all)


# TO HIDE PYPLOT WARNING

elif (rad == "PLAYER COMPARISON"):

    st.title("PLAYER COMPARISON")
    player_dict = players.get_players()
    # Conversion Of Dictionary To Dataframe
    player_data = pd.DataFrame.from_dict(player_dict)
    player_info1 = st.sidebar.selectbox("Player - 1 ", player_data['full_name'], key="Pl1", index =871 )#871 - Stephen Curry Index  1946- Lebron
    player_info2 = st.sidebar.selectbox("Player - 2 ", player_data['full_name'],key="Pl2", index=1946)

    selected_player1 = [player for player in player_dict if player['full_name'] == player_info1][0]
    id_select1 = selected_player1['id']

    selected_player2 = [player for player in player_dict if player['full_name'] == player_info2][0]
    id_select2 = selected_player2['id']

    stat_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))))

    gamelog_player1 = playergamelog.PlayerGameLog(player_id=id_select1, season=stat_year)
    gamelog_player2 = playergamelog.PlayerGameLog(player_id=id_select2, season=stat_year)
    df_gamelog1 = gamelog_player1.get_data_frames()[0]
    df_gamelog2 = gamelog_player2.get_data_frames()[0]

    avg_points_p1 = int(df_gamelog1['PTS'].mean())
    avg_points_p2 = int(df_gamelog2['PTS'].mean())

    avg_assists_p1 = int(df_gamelog1['AST'].mean())
    avg_assists_p2 = int(df_gamelog2['AST'].mean())

    avg_rebounds_p1 = int(df_gamelog1['REB'].mean())
    avg_rebounds_p2 = int(df_gamelog2['REB'].mean())

    avg_blocks_p1 = int(df_gamelog1['BLK'].mean())
    avg_blocks_p2 = int(df_gamelog2['BLK'].mean())

    avg_steals_p1 = int(df_gamelog1['STL'].mean())
    avg_steals_p2 = int(df_gamelog2['STL'].mean())

    avg_turnovers_p1 = int(df_gamelog1['TOV'].mean())
    avg_turnovers_p2 = int(df_gamelog2['TOV'].mean())

    # performance_index

    b1 = st.button("Season Average Record")
    if(b1):
        Avg_Points = {player_info1: avg_points_p1, player_info2: avg_points_p2}
        Avg_Assists = {player_info1: avg_assists_p1, player_info2: avg_assists_p2}
        Avg_Rebounds = {player_info1: avg_rebounds_p1, player_info2: avg_rebounds_p2}
        Avg_Blocks = {player_info1: avg_blocks_p1, player_info2: avg_blocks_p2}
        Avg_Steals = {player_info1: avg_steals_p1, player_info2: avg_steals_p2}
        Avg_Turnovers = {player_info1: avg_turnovers_p1, player_info2: avg_turnovers_p2}


        player_data = [Avg_Points, Avg_Assists, Avg_Rebounds, Avg_Blocks, Avg_Steals, Avg_Turnovers]
        player_index = ['Points', 'Assists', 'Rebounds', 'Blocks', 'Steals', 'Turnovers']

        comparison_df = pd.DataFrame(data=player_data, index= player_index)
        st.write(comparison_df)



    b2 = st.button("Graphical Comparison")
    if(b2):


        Graph_df = pd.DataFrame([[avg_points_p1, player_info1, 'PTS'],
                                 [avg_points_p2, player_info2, 'PTS'],
                                 [avg_assists_p1, player_info1, 'AST'],
                                 [avg_assists_p2, player_info2, 'AST'],
                                 [avg_rebounds_p1, player_info1, 'REBS'],
                                 [avg_rebounds_p2, player_info2, 'REBS'],
                                 [avg_blocks_p1, player_info1, 'BLK'],
                                 [avg_blocks_p2, player_info2, 'BLK'],
                                 [avg_steals_p1, player_info1, 'STL'],
                                 [avg_steals_p2, player_info2, 'STL'],
                                 [avg_turnovers_p1, player_info1, 'TOV'],
                                 [avg_turnovers_p2, player_info2, 'TOV']],
                                 columns=['AVG_STATS', 'PLAYER', 'CATEGORY'])

        stats_chart = alt.Chart(Graph_df).mark_bar().encode(
            alt.Column('CATEGORY'), alt.X('PLAYER'),
            alt.Y('AVG_STATS', axis=alt.Axis(grid= False)),
            alt.Color('PLAYER')
        )
        st.altair_chart(stats_chart)


elif(rad == "PERFORMANCE INDEX"):
    st.title('PERFORMANCE INDEX')

    player_dict = players.get_players()
    # Conversion Of Dictionary To Dataframe
    player_data = pd.DataFrame.from_dict(player_dict)
    player_info = st.sidebar.selectbox("Player Name", player_data['full_name'], key="Pl",index=871)  #871 - Stephen Curry Index  1946- Lebron

    selected_player = [player for player in player_dict if player['full_name'] == player_info][0]
    id_select = selected_player['id']

    stat_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))))

    gamelog_player = playergamelog.PlayerGameLog(player_id=id_select, season=stat_year)
    df_gamelog = gamelog_player.get_data_frames()[0]

    avg_points = int(df_gamelog['PTS'].mean())
    avg_assists = int(df_gamelog['AST'].mean())
    avg_rebounds = int(df_gamelog['REB'].mean())
    avg_blocks = int(df_gamelog['BLK'].mean())
    avg_steals = int(df_gamelog['STL'].mean())
    avg_turnovers = int(df_gamelog['TOV'].mean())
    avg_madeFG = int(df_gamelog['FGM'].mean())
    avg_attemptedFG = int(df_gamelog['FGA'].mean())
    avg_missedFG = avg_attemptedFG - avg_madeFG
    plus_minus = int(df_gamelog['PLUS_MINUS'].mean())


    b1 = st.button("Season Average Record")
    if (b1):
        Avg_Points = {player_info: avg_points}
        Avg_Assists = {player_info: avg_assists}
        Avg_Rebounds = {player_info: avg_rebounds}
        Avg_Blocks = {player_info: avg_blocks}
        Avg_Steals = {player_info: avg_steals}
        Avg_Turnovers = {player_info: avg_turnovers}

        player_data = [Avg_Points, Avg_Assists, Avg_Rebounds, Avg_Blocks, Avg_Steals, Avg_Turnovers]
        player_index = ['Points', 'Assists', 'Rebounds', 'Blocks', 'Steals', 'Turnovers']

        comparison_df = pd.DataFrame(data=player_data, index=player_index)
        st.write(comparison_df)

    b2 = st.button("Performance Index Calculation")
    if(b2):
        PI = avg_points + avg_assists + avg_rebounds + avg_steals - avg_missedFG - plus_minus

        st.write(PI)

elif (rad == "LOVE FOR THE GAME"):
    st.title("ICONIC MOMENTS")
    hist_url = 'https://www.nba.com/history'
    b1 = st.button("History")
    if (b1):
        webbrowser.open_new(hist_url)

    b2 = st.button("Trailer")
    if(b2):
        st.video("Teaser.mp4")

    b3 = st.button("Iconic")
    if (b3):
        st.image("jordan.jpg")

st.set_option('deprecation.showPyplotGlobalUse', False)