-- EPL Team Stats Database Schema
-- Stores current statistics for all Premier League teams

CREATE TABLE IF NOT EXISTS team_stats (
    team_name VARCHAR(100) PRIMARY KEY,

    -- ELO rating
    elo_rating FLOAT NOT NULL DEFAULT 1500,

    -- Rolling averages (last 5 matches)
    goals_for_avg FLOAT NOT NULL DEFAULT 1.5,
    goals_against_avg FLOAT NOT NULL DEFAULT 1.0,
    points_avg FLOAT NOT NULL DEFAULT 1.5,

    -- Match tracking
    last_match_date DATE,
    matches_played INT NOT NULL DEFAULT 0,

    -- Metadata
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    season VARCHAR(20) NOT NULL DEFAULT '2024-25'
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_team_name ON team_stats(team_name);
CREATE INDEX IF NOT EXISTS idx_updated_at ON team_stats(updated_at);

-- Comments for documentation
COMMENT ON TABLE team_stats IS 'Current statistics for EPL teams used for match predictions';
COMMENT ON COLUMN team_stats.elo_rating IS 'Current ELO rating (chess-style rating system)';
COMMENT ON COLUMN team_stats.goals_for_avg IS 'Average goals scored in last 5 matches';
COMMENT ON COLUMN team_stats.goals_against_avg IS 'Average goals conceded in last 5 matches';
COMMENT ON COLUMN team_stats.points_avg IS 'Average points earned in last 5 matches';
COMMENT ON COLUMN team_stats.last_match_date IS 'Date of the team''s most recent match';
