function handles = show_graph_2D_FastRender(step,fighandle,r,bound,N,A,contour_x,contour_y,contour_levels,handles,trail)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Display Settings %%%%%%%%%%%%%%%%%
trail_on = true;
trail_color = [1.0, 0.0, 0.0, 0.2]; % [Red, Green, Blue, Opacity]
ghost_interval = 0;
ghost_line_width = 2;
ghost_graph_lines_color = [0,0,1,0.35];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

visgraph = ones(2,N*length(A))*NaN;
visidx = 1;
for i=1:N
    I=find(A(i,:));   
    for j=1:length(I)
        visgraph(1,visidx) = r(i,1);
        visgraph(2,visidx) = r(i,2);
        visidx = visidx + 1;
        visgraph(1,visidx) = r(I(j),1);
        visgraph(2,visidx) = r(I(j),2);
        visidx = visidx + 1;
        visgraph(1,visidx) = NaN;
        visgraph(2,visidx) = NaN;
        visidx = visidx + 1;
    end
end

set(0, 'CurrentFigure', fighandle)

if isempty(fieldnames(handles))
    grid on
    hold on
    h_trail = line(trail.x,trail.y,'color',trail_color,'LineWidth',ghost_line_width);
    
    edgecolors = contour_levels;
    patch(contour_x,contour_y,edgecolors,'EdgeColor','flat');
    
    ghosts.loc.x = r(:,1);
    ghosts.loc.y = r(:,2);
%     ghosts.directions.x = n2(:,1)/3;
%     ghosts.directions.y = n2(:,2)/3;
    ghosts.graph.x = visgraph(1,:);
    ghosts.graph.y = visgraph(2,:);
    ghosts.markers_handle = plot(ghosts.loc.x,ghosts.loc.y,'o','color','b','MarkerSize',4,'MarkerFaceColor','b');
    %ghosts.quiver_handle = quiver(ghosts.loc.x,ghosts.loc.y,...
     %   ghosts.directions.x,ghosts.directions.y,0,'b','LineWidth',1);
    ghosts.graph_handle = line(ghosts.graph.x,ghosts.graph.y,'color',ghost_graph_lines_color,'LineWidth',0.5);
    
    h_mar = plot(r(:,1),r(:,2),'o','color','r','MarkerSize',8,'MarkerFaceColor','r');
   % h_quiv = quiver(r(:,1),r(:,2),n2(:,1)/6,n2(:,2)/6,0,'r','LineWidth',2);
    h_lin = line(visgraph(1,:),visgraph(2,:),'color','b','LineWidth',0.5);
    hold off
    axis([bound(1),bound(2),bound(3),bound(4)]);
    
    handles = struct('mar',h_mar,'lin',h_lin,'trail',h_trail,'ghosts',ghosts);
else
    %set(handles.quiv,'XData',r(:,1),'YData',r(:,2),'UData',n2(:,1)/6,'VData',n2(:,2)/6);
    set(handles.mar,'XData',r(:,1),'YData',r(:,2));
    set(handles.lin,'XData',visgraph(1,:),'YData',visgraph(2,:));

    if trail_on
        newX = num2cell(trail.x',2)';
        newY = num2cell(trail.y',2)';
        [handles.trail.XData] = newX{:};
        [handles.trail.YData] = newY{:};
    end
    
    if mod(step,ghost_interval) == 0
        handles.ghosts.loc.x = [handles.ghosts.loc.x; r(:,1)];
        handles.ghosts.loc.y = [handles.ghosts.loc.y; r(:,2)];
        set(handles.ghosts.markers_handle,'XData',handles.ghosts.loc.x,'YData',handles.ghosts.loc.y);
        
%         handles.ghosts.directions.x = [handles.ghosts.directions.x; n2(:,1)/3];
%         handles.ghosts.directions.y = [handles.ghosts.directions.y; n2(:,2)/3];
%         set(handles.ghosts.quiver_handle,'XData',handles.ghosts.loc.x,'YData',handles.ghosts.loc.y,...
%             'UData',handles.ghosts.directions.x,'VData',handles.ghosts.directions.y);
%         
        handles.ghosts.graph.x = [handles.ghosts.graph.x, NaN, visgraph(1,:)];
        handles.ghosts.graph.y = [handles.ghosts.graph.y, NaN, visgraph(2,:)];
        set(handles.ghosts.graph_handle,'XData',handles.ghosts.graph.x,'YData',handles.ghosts.graph.y);
    end
end

drawnow;


