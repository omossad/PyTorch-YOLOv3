[x,y] = read_files('nhl');
x = x/0.125;
y = y/0.125;
draw_heatmap(x,y);