


vmaf_base = [98.5572, 96.1635, 89.3398, 77.5776];
vmaf_base_rate =  [7.6615, 3.9542, 1.8433, 0.7933];

vmaf_lambda_r = [98.4513, 96.1780, 89.0352, 76.9711];
vmaf_lambda_r_rate = [7.7132, 4.0258, 1.8357, 0.7934];

vmaf_roi = [98.3875, 95.2929, 87.2638, 74.9503];
vmaf_roi_rate = [7.7604, 4.0047, 1.8435, 0.7938];


% Defaults for this blog post
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

figure(2);
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties
plot(vmaf_base_rate, vmaf_base,'-.b^','LineWidth',lw,'MarkerSize',msz);
hold on 
plot(vmaf_lambda_r_rate,vmaf_lambda_r,'-ro','LineWidth',lw,'MarkerSize',msz);
hold on 
plot(vmaf_roi_rate,vmaf_roi,':kx','LineWidth',lw,'MarkerSize',msz);

xlim([0 8]);
legend('Base', 'CAVE', 'DeepGame','location', 'northoutside');
xlabel('Bitrate (Mbps)');
title('VMAF');


% Set Tick Marks
set(gca,'XTick',0:8);
set(gca,'YTick',40:20:100);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

% Save the file as PNG
print('improvedExample','-dpng','-r300');

print('improvedExample','-depsc2','-r300');
if ispc % Use Windows ghostscript call
  system('gswin64c -o -q -sDEVICE=png256 -dEPSCrop -r300 -oimprovedExample_eps.png improvedExample.eps');
else % Use Unix/OSX ghostscript call
  system('gs -o -q -sDEVICE=png256 -dEPSCrop -r300 -oimprovedExample_eps.png improvedExample.eps');
end
