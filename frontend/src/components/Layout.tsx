import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Divider,
  Avatar,
  Menu,
  MenuItem,
  Badge,
  Tooltip,
  Chip,
  Alert,
  Collapse,
  Breadcrumbs,
  Link,
  useMediaQuery,
  useTheme,
  Zoom,
  Fab,
} from '@mui/material';
import {
  MenuOutlined,
  DarkModeOutlined,
  LightModeOutlined,
  NotificationsOutlined,
  AccountCircleOutlined,
  DashboardOutlined,
  PredictionsOutlined,
  DatasetOutlined,
  ModelTrainingOutlined,
  AnalyticsOutlined,
  SettingsOutlined,
  HelpOutlineOutlined,
  LogoutOutlined,
  HomeOutlined,
  ExpandLessOutlined,
  ExpandMoreOutlined,
  HealthAndSafetyOutlined,
  KeyboardArrowUpOutlined,
  CircleOutlined,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

// Types and hooks
import { ThemeMode, NavigationItem, BreadcrumbItem } from '../types';
import { useAuth } from '../hooks/useAuth';
import { useNotifications } from '../hooks/useNotifications';
import { useHealthCheck } from '../hooks/useHealthCheck';

// Components
import LoadingSpinner from './common/LoadingSpinner';

// Constants
const DRAWER_WIDTH = 280;
const MINI_DRAWER_WIDTH = 64;

interface LayoutProps {
  children: React.ReactNode;
  themeMode: ThemeMode;
  onThemeToggle: () => void;
}

const Layout: React.FC<LayoutProps> = ({ children, themeMode, onThemeToggle }) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // State
  const [mobileOpen, setMobileOpen] = useState(false);
  const [drawerMinimized, setDrawerMinimized] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [notificationMenuAnchor, setNotificationMenuAnchor] = useState<null | HTMLElement>(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [expandedNavItems, setExpandedNavItems] = useState<string[]>(['predictions']);

  // Hooks
  const { user, logout, isAuthenticated } = useAuth();
  const { notifications, markAsRead, clearAll } = useNotifications();
  const { healthStatus } = useHealthCheck();

  // Navigation items configuration
  const navigationItems: NavigationItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      path: '/',
      icon: DashboardOutlined,
      description: 'Overview and quick access',
    },
    {
      id: 'predictions',
      label: 'Predictions',
      path: '/predict',
      icon: PredictionsOutlined,
      description: 'Cardiovascular risk assessment',
      children: [
        {
          id: 'single-prediction',
          label: 'Single Prediction',
          path: '/predict',
          icon: CircleOutlined,
          description: 'Individual patient assessment',
        },
        {
          id: 'batch-prediction',
          label: 'Batch Prediction',
          path: '/predict/batch',
          icon: CircleOutlined,
          description: 'Multiple patient assessments',
        },
      ],
    },
    {
      id: 'data',
      label: 'Data Management',
      path: '/data',
      icon: DatasetOutlined,
      description: 'Dataset upload and management',
    },
    {
      id: 'models',
      label: 'Model Management',
      path: '/models',
      icon: ModelTrainingOutlined,
      description: 'ML model training and monitoring',
    },
    {
      id: 'analytics',
      label: 'Analytics',
      path: '/analytics',
      icon: AnalyticsOutlined,
      description: 'Performance metrics and insights',
    },
    {
      id: 'settings',
      label: 'Settings',
      path: '/settings',
      icon: SettingsOutlined,
      description: 'Application configuration',
    },
  ];

  // Scroll to top handler
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Generate breadcrumbs from current path
  const breadcrumbs: BreadcrumbItem[] = useMemo(() => {
    const pathSegments = location.pathname.split('/').filter(Boolean);
    const crumbs: BreadcrumbItem[] = [{ label: 'Home', path: '/', icon: HomeOutlined }];

    let currentPath = '';
    for (const segment of pathSegments) {
      currentPath += `/${segment}`;
      
      // Find navigation item for this path
      const findNavItem = (items: NavigationItem[]): NavigationItem | undefined => {
        for (const item of items) {
          if (item.path === currentPath) return item;
          if (item.children) {
            const found = findNavItem(item.children);
            if (found) return found;
          }
        }
      };

      const navItem = findNavItem(navigationItems);
      
      crumbs.push({
        label: navItem?.label || segment.charAt(0).toUpperCase() + segment.slice(1),
        path: currentPath,
        icon: navItem?.icon,
      });
    }

    return crumbs;
  }, [location.pathname]);

  // Handle navigation
  const handleNavigate = useCallback((path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileOpen(false);
    }
  }, [navigate, isMobile]);

  // Handle drawer toggle
  const handleDrawerToggle = useCallback(() => {
    if (isMobile) {
      setMobileOpen(!mobileOpen);
    } else {
      setDrawerMinimized(!drawerMinimized);
    }
  }, [isMobile, mobileOpen, drawerMinimized]);

  // Handle navigation item expansion
  const handleNavItemToggle = useCallback((itemId: string) => {
    setExpandedNavItems(prev =>
      prev.includes(itemId)
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId]
    );
  }, []);

  // Handle user menu
  const handleUserMenuOpen = useCallback((event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  }, []);

  const handleUserMenuClose = useCallback(() => {
    setUserMenuAnchor(null);
  }, []);

  // Handle logout
  const handleLogout = useCallback(async () => {
    await logout();
    handleUserMenuClose();
    navigate('/');
  }, [logout, navigate]);

  // Scroll to top
  const scrollToTop = useCallback(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  // Render navigation item
  const renderNavigationItem = (item: NavigationItem, level = 0) => {
    const isActive = location.pathname === item.path;
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expandedNavItems.includes(item.id);
    const ItemIcon = item.icon;

    return (
      <React.Fragment key={item.id}>
        <ListItem disablePadding sx={{ display: 'block' }}>
          <ListItemButton
            onClick={() => {
              if (hasChildren) {
                handleNavItemToggle(item.id);
              } else {
                handleNavigate(item.path);
              }
            }}
            selected={isActive}
            sx={{
              minHeight: 48,
              justifyContent: drawerMinimized && !isMobile ? 'center' : 'flex-start',
              px: 2.5,
              ml: level * 2,
              borderRadius: 2,
              mx: 1,
              mb: 0.5,
              '&.Mui-selected': {
                backgroundColor: 'primary.main',
                color: 'primary.contrastText',
                '&:hover': {
                  backgroundColor: 'primary.dark',
                },
                '& .MuiListItemIcon-root': {
                  color: 'primary.contrastText',
                },
              },
              '&:hover': {
                backgroundColor: 'action.hover',
                borderRadius: 2,
              },
            }}
          >
            <ListItemIcon
              sx={{
                minWidth: 0,
                mr: drawerMinimized && !isMobile ? 0 : 3,
                justifyContent: 'center',
                color: isActive ? 'inherit' : 'text.secondary',
              }}
            >
              {ItemIcon && <ItemIcon />}
            </ListItemIcon>
            
            {(!drawerMinimized || isMobile) && (
              <ListItemText
                primary={item.label}
                secondary={level === 0 ? item.description : undefined}
                primaryTypographyProps={{
                  fontSize: level === 0 ? '0.875rem' : '0.8125rem',
                  fontWeight: isActive ? 600 : 500,
                }}
                secondaryTypographyProps={{
                  fontSize: '0.75rem',
                  noWrap: true,
                }}
              />
            )}

            {hasChildren && (!drawerMinimized || isMobile) && (
              <IconButton size="small" sx={{ color: 'inherit' }}>
                {isExpanded ? <ExpandLessOutlined /> : <ExpandMoreOutlined />}
              </IconButton>
            )}

            {item.badge && (!drawerMinimized || isMobile) && (
              <Chip
                label={item.badge}
                size="small"
                color="secondary"
                sx={{ fontSize: '0.6875rem', height: 20 }}
              />
            )}
          </ListItemButton>
        </ListItem>

        {/* Render children */}
        {hasChildren && (!drawerMinimized || isMobile) && (
          <Collapse in={isExpanded} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {item.children!.map(child => renderNavigationItem(child, level + 1))}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  // Render drawer content
  const renderDrawerContent = () => (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo and branding */}
      <Box sx={{ 
        p: 2, 
        borderBottom: '1px solid', 
        borderColor: 'divider',
        minHeight: 64,
        display: 'flex',
        alignItems: 'center',
        gap: 2
      }}>
        <Avatar sx={{ bgcolor: 'primary.main', width: 40, height: 40 }}>
          <HealthAndSafetyOutlined />
        </Avatar>
        
        {(!drawerMinimized || isMobile) && (
          <Box sx={{ flexGrow: 1, minWidth: 0 }}>
            <Typography variant="h6" noWrap sx={{ fontWeight: 700, fontSize: '1rem' }}>
              CVD Prediction
            </Typography>
            <Typography variant="body2" color="text.secondary" noWrap sx={{ fontSize: '0.75rem' }}>
              Healthcare ML System
            </Typography>
          </Box>
        )}
      </Box>

      {/* Navigation */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', py: 1 }}>
        <List>
          {navigationItems.map(item => renderNavigationItem(item))}
        </List>
      </Box>

      {/* System health indicator */}
      {healthStatus && (!drawerMinimized || isMobile) && (
        <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'divider' }}>
          <Alert 
            severity={
              healthStatus.overall_status === 'healthy' ? 'success' :
              healthStatus.overall_status === 'degraded' ? 'warning' : 'error'
            }
            variant="outlined"
            sx={{ 
              fontSize: '0.75rem',
              '& .MuiAlert-message': { fontSize: '0.75rem' }
            }}
          >
            System Status: {healthStatus.overall_status}
          </Alert>
        </Box>
      )}

      {/* Version info */}
      {(!drawerMinimized || isMobile) && (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            v1.0.0 • {process.env.REACT_APP_ENVIRONMENT}
          </Typography>
        </Box>
      )}
    </Box>
  );

  // Render app bar content
  const renderAppBar = () => (
    <AppBar 
      position="fixed" 
      elevation={0}
      sx={{ 
        width: { sm: `calc(100% - ${isMobile ? 0 : drawerMinimized ? MINI_DRAWER_WIDTH : DRAWER_WIDTH}px)` },
        ml: { sm: `${isMobile ? 0 : drawerMinimized ? MINI_DRAWER_WIDTH : DRAWER_WIDTH}px` },
        borderBottom: '1px solid',
        borderColor: 'divider',
        backgroundColor: 'background.paper',
        color: 'text.primary',
      }}
    >
      <Toolbar sx={{ gap: 2 }}>
        {/* Menu toggle */}
        <IconButton
          color="inherit"
          aria-label="toggle navigation"
          edge="start"
          onClick={handleDrawerToggle}
          sx={{ mr: 2 }}
        >
          <MenuOutlined />
        </IconButton>

        {/* Breadcrumbs */}
        <Box sx={{ flexGrow: 1 }}>
          <Breadcrumbs aria-label="navigation breadcrumb">
            {breadcrumbs.map((crumb, index) => {
              const isLast = index === breadcrumbs.length - 1;
              const CrumbIcon = crumb.icon;

              return isLast ? (
                <Typography 
                  key={crumb.path} 
                  color="text.primary" 
                  sx={{ display: 'flex', alignItems: 'center', gap: 0.5, fontWeight: 600 }}
                >
                  {CrumbIcon && <CrumbIcon sx={{ fontSize: 16 }} />}
                  {crumb.label}
                </Typography>
              ) : (
                <Link
                  key={crumb.path}
                  underline="hover"
                  color="inherit"
                  href={crumb.path}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(crumb.path!);
                  }}
                  sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}
                >
                  {CrumbIcon && <CrumbIcon sx={{ fontSize: 16 }} />}
                  {crumb.label}
                </Link>
              );
            })}
          </Breadcrumbs>
        </Box>

        {/* Action buttons */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Theme toggle */}
          <Tooltip title={`Switch to ${themeMode === 'light' ? 'dark' : 'light'} mode`}>
            <IconButton color="inherit" onClick={onThemeToggle}>
              {themeMode === 'light' ? <DarkModeOutlined /> : <LightModeOutlined />}
            </IconButton>
          </Tooltip>

          {/* Notifications */}
          <Tooltip title="Notifications">
            <IconButton
              color="inherit"
              onClick={(e) => setNotificationMenuAnchor(e.currentTarget)}
            >
              <Badge badgeContent={notifications.length} color="error">
                <NotificationsOutlined />
              </Badge>
            </IconButton>
          </Tooltip>

          {/* User menu */}
          {isAuthenticated && user && (
            <Tooltip title="User menu">
              <IconButton
                color="inherit"
                onClick={handleUserMenuOpen}
                sx={{ p: 0.5 }}
              >
                <Avatar 
                  sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}
                  alt={user.full_name || user.username}
                >
                  {(user.full_name || user.username).charAt(0).toUpperCase()}
                </Avatar>
              </IconButton>
            </Tooltip>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );

  // Render user menu
  const renderUserMenu = () => (
    <Menu
      anchorEl={userMenuAnchor}
      open={Boolean(userMenuAnchor)}
      onClose={handleUserMenuClose}
      onClick={handleUserMenuClose}
      PaperProps={{
        elevation: 8,
        sx: {
          mt: 1.5,
          minWidth: 220,
          '& .MuiMenuItem-root': {
            px: 2,
            py: 1.5,
          },
        },
      }}
      transformOrigin={{ horizontal: 'right', vertical: 'top' }}
      anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
    >
      {user && (
        <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
          <Typography variant="subtitle1" fontWeight={600}>
            {user.full_name || user.username}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {user.email}
          </Typography>
          {user.roles && user.roles.length > 0 && (
            <Box sx={{ mt: 1, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
              {user.roles.slice(0, 2).map(role => (
                <Chip
                  key={role}
                  label={role}
                  size="small"
                  variant="outlined"
                  sx={{ fontSize: '0.6875rem', height: 20 }}
                />
              ))}
            </Box>
          )}
        </Box>
      )}

      <MenuItem onClick={() => navigate('/settings')}>
        <ListItemIcon>
          <SettingsOutlined fontSize="small" />
        </ListItemIcon>
        <ListItemText>Settings</ListItemText>
      </MenuItem>

      <MenuItem onClick={() => navigate('/docs')}>
        <ListItemIcon>
          <HelpOutlineOutlined fontSize="small" />
        </ListItemIcon>
        <ListItemText>Help & Documentation</ListItemText>
      </MenuItem>

      <Divider />

      <MenuItem onClick={handleLogout}>
        <ListItemIcon>
          <LogoutOutlined fontSize="small" />
        </ListItemIcon>
        <ListItemText>Logout</ListItemText>
      </MenuItem>
    </Menu>
  );

  // Render notifications menu
  const renderNotificationsMenu = () => (
    <Menu
      anchorEl={notificationMenuAnchor}
      open={Boolean(notificationMenuAnchor)}
      onClose={() => setNotificationMenuAnchor(null)}
      PaperProps={{
        elevation: 8,
        sx: {
          mt: 1.5,
          maxWidth: 360,
          maxHeight: 400,
        },
      }}
      transformOrigin={{ horizontal: 'right', vertical: 'top' }}
      anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
    >
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6" fontSize="1rem">
            Notifications
          </Typography>
          {notifications.length > 0 && (
            <Typography
              variant="body2"
              color="primary"
              sx={{ cursor: 'pointer' }}
              onClick={clearAll}
            >
              Clear All
            </Typography>
          )}
        </Box>
      </Box>

      {notifications.length === 0 ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            No new notifications
          </Typography>
        </Box>
      ) : (
        <Box sx={{ maxHeight: 300, overflow: 'auto' }}>
          {notifications.slice(0, 5).map((notification) => (
            <MenuItem
              key={notification.id}
              onClick={() => markAsRead(notification.id)}
              sx={{ 
                whiteSpace: 'normal',
                alignItems: 'flex-start',
                py: 1.5,
              }}
            >
              <Box>
                <Typography variant="subtitle2" fontSize="0.875rem">
                  {notification.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" fontSize="0.75rem">
                  {notification.message}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {notification.timestamp.toLocaleTimeString()}
                </Typography>
              </Box>
            </MenuItem>
          ))}
        </Box>
      )}
    </Menu>
  );

  const drawerWidth = drawerMinimized && !isMobile ? MINI_DRAWER_WIDTH : DRAWER_WIDTH;

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      {/* App Bar */}
      {renderAppBar()}

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: DRAWER_WIDTH,
              backgroundColor: 'background.paper',
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
        >
          {renderDrawerContent()}
        </Drawer>

        {/* Desktop drawer */}
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              backgroundColor: 'background.paper',
              borderRight: '1px solid',
              borderColor: 'divider',
              transition: theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
              }),
            },
          }}
          open
        >
          {renderDrawerContent()}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 0,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Toolbar spacer */}
        <Toolbar />

        {/* Page content */}
        <Box sx={{ flexGrow: 1, position: 'relative' }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3 }}
              style={{ height: '100%' }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </Box>

        {/* Footer */}
        <Box
          component="footer"
          sx={{
            py: 2,
            px: 3,
            mt: 'auto',
            borderTop: '1px solid',
            borderColor: 'divider',
            backgroundColor: 'background.paper',
          }}
        >
          <Typography variant="body2" color="text.secondary" align="center">
            © 2024 Cardiovascular Disease Prediction System • 
            Built with ❤️ for Healthcare • 
            <Link 
              href="/docs" 
              underline="hover" 
              sx={{ ml: 1 }}
              onClick={(e) => {
                e.preventDefault();
                navigate('/docs');
              }}
            >
              Documentation
            </Link>
          </Typography>
        </Box>
      </Box>

      {/* User Menu */}
      {renderUserMenu()}

      {/* Notifications Menu */}
      {renderNotificationsMenu()}

      {/* Scroll to top button */}
      <AnimatePresence>
        {showScrollTop && (
          <motion.div
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'fixed',
              bottom: 24,
              right: 24,
              zIndex: 1000,
            }}
          >
            <Zoom in={showScrollTop}>
              <Fab
                color="primary"
                size="medium"
                onClick={scrollToTop}
                aria-label="scroll to top"
              >
                <KeyboardArrowUpOutlined />
              </Fab>
            </Zoom>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Skip navigation link for accessibility */}
      <Link
        href="#main-content"
        sx={{
          position: 'absolute',
          top: -40,
          left: 6,
          zIndex: 9999,
          backgroundColor: 'primary.main',
          color: 'primary.contrastText',
          padding: '8px 16px',
          textDecoration: 'none',
          borderRadius: 1,
          '&:focus': {
            top: 6,
          },
        }}
      >
        Skip to main content
      </Link>
    </Box>
  );
};

export default Layout;