import React from "react";
import { Route, Redirect } from "react-router-dom";
import _ from 'lodash';
import Login from "./Pages/Login/login";
import Proceed from './Pages/Proceed/proceed';

import PosDashboard from './Pages/Dashboard/PosDashboard/PosDashboard';
import GeographyInReport from './Pages/GeographycalDemography/GeographycalDemographyChart';
import RouteDashboard from './Pages/Dashboard/RouteDashboard/RouteDashboard';
import DemographyDashboard from "./Pages/Dashboard/DemographyDashboard/DemographyDashboard";
import AgentDashboard from './Pages/Dashboard/AgentDashboard/AgentDashboard';
import RouteRevenuePlanning from './Pages/Dashboard/RouteRevenuePlanning/RouteRevenuePlanning';
import RouteProfitability from './Pages/Dashboard/RouteProfitability/RouteProfitability';
import PosRevenuePlanning from './Pages/Dashboard/PosRevenuePlanning/PosRevenuePlanning';

import POSDetail from "./Pages/PosDetails/posDetails";
import DemographyAgeGroup from "./Pages/DemographyAgeGroup/demographyAgeGroup";
import posPromotionTracking from './Pages/POSPromotionTracking/posPromotionTracking'
import Routes from "./Pages/Route/route";
import TopMarkets from './Pages/TopMarkets/topMarkets';
import CompetitorAnalysis from './Pages/CompetitorAnalysis/competitorAnalysis';
import AgentAnalysis from './Pages/AgentAnalysis/agentAnalysis';
import ChannelPerformance from './Pages/ChannelPerformance/channelPerformance';
import SubChannelPerformance from './Pages/SubChannelPerformance/subChannelPerformance';
import Segmentation from './Pages/Segmentation/segmentation';
import SubSegmentation from './Pages/SubSegmentation/subSegmentation';
import DemoReport from './Pages/DemographyReportPage/demographyReportPage';
import DDSChart from './Pages/DDSChart/DDSChart';
import Interline from './Pages/Interline/InterlineReport'
import DataLoadIndicator from './Pages/DataLoadIndicator/DataLoadIndicator';
import RPSSelection from "./Pages/RPSValueSelection/RPSSelectionPage"
import RPSPos from "./Pages/RPS/RPSPos";
import RPSRoute from "./Pages/RPS/RPSRoute";
import ForecastAccuracy from "./Pages/ForecastAccuracy/forecastAccuracy";
import RouteProfitabilityConsolidate from "./Pages/RouteProfitabilityConsolidate/routeProfitabilityConsolidate";
import RouteProfitabiltySolutions from "./Pages/RouteProfitabilitySolution/routeProfitabilitySolution";
import SystemModuleSearch from "./Pages/SystemModule/SystemModuleSearch";
import SystemModuleEdit from "./Pages/SystemModule/SystemModuleEdit";
import SystemModuleAdd from "./Pages/SystemModule/SystemModuleAdd";
import SearchUser from "./Pages/UserManagement/SearchUser";
import CreateUser from "./Pages/UserManagement/CreateUser";
import EditUser from "./Pages/UserManagement/EditUser";
import UserProfile from "./Pages/UserManagement/UserProfile";
import ChangePassword from "./Pages/UserManagement/ChangePassword";
import AuditLogs from "./Pages/AuditLogs/AuditLogs";
import Alerts from "./Pages/Alerts/Alerts";
import AlertSetting from "./Pages/Alerts/AlertSetting";
import RPSSettings from "./Pages/RPS/RPSSettings";
import TargetRational from "./Pages/TargetRational/TargetRational";
import InvalidLogin from "./Pages/InvalidLogin/InvalidLogin";
import cookieStorage from "./Constants/cookie-storage";
import PageNotFound from "./Pages/PageNotFound/PageNotFound";
import SiteUnderMaintenance from "./Pages/SiteUnderMaintenance/SiteUnderMaintenance";
import Test from './Component/test';
import config from './Constants/config'
import AlertsNew from './Pages/AlertsNewModule/alertsnew'
import AlertDetails from "./Pages/AlertsNewModule/alertsDetails";

const PrivateRoute = ({ component: Component, ...rest }) => (
  <Route {...rest} render={(props) => (
    !_.isEmpty(cookieStorage.getCookie('Authorization'))
      ? (<>
        <Component {...props} />
        <p className={`copyright-message ${rest.path.toLowerCase().includes('dashboard') ? "grow" : ""} `}>Copyright © 2022 Revemax | All rights reserved | Contact <a href="http://www.revemax.ai">www.revemax.ai</a></p>
      </>)
      : <Redirect to='/' />
  )} />
)

const BaseRouter = () => (
  < div >
    {/* <Route path="/test123" component={Test} /> */}
    <Route exact path="/" component={config.UNDER_MAINTENANCE ? SiteUnderMaintenance : Login} />
    <Route path='/proceed' component={Proceed} />
    <PrivateRoute path="/posDashboard" component={PosDashboard} />
    <PrivateRoute path="/routeDashboard" component={RouteDashboard} />
    <PrivateRoute path="/demographyDashboard" component={DemographyDashboard} />
    <PrivateRoute path="/decisionMakingDashboard" component={AgentDashboard} />
    <PrivateRoute path="/routeRevenuePlanning" component={RouteRevenuePlanning} />
    <PrivateRoute path="/routeProfitability" component={RouteProfitability} />
    <PrivateRoute path="/posRevenuePlanning" component={PosRevenuePlanning} />
    <PrivateRoute path="/systemModuleSearch" component={SystemModuleSearch} />
    <PrivateRoute path="/systemModuleAdd" component={SystemModuleAdd} />
    <PrivateRoute path="/systemModuleEdit" component={SystemModuleEdit} />
    <PrivateRoute path="/userProfile" component={UserProfile} />
    <PrivateRoute path="/createUser" component={CreateUser} />
    <PrivateRoute path="/searchUser" component={SearchUser} />
    <PrivateRoute path="/editUser" component={EditUser} />
    <PrivateRoute path="/changePassword" component={ChangePassword} />
    <PrivateRoute path="/pos" component={POSDetail} />
    <PrivateRoute path="/route" component={Routes} />
    <PrivateRoute path="/demographyAgeGroup" component={DemographyAgeGroup} />
    <PrivateRoute path="/geographyInReport" component={GeographyInReport} />
    <PrivateRoute path="/demographicReport" component={DemoReport} />
    <PrivateRoute path="/posPromotion" component={posPromotionTracking} />
    <PrivateRoute path="/topMarkets" component={TopMarkets} />
    <PrivateRoute path="/competitorAnalysis" component={CompetitorAnalysis} />
    <PrivateRoute path="/agentAnalysis" component={AgentAnalysis} />
    <PrivateRoute path="/channelPerformance" component={ChannelPerformance} />
    <PrivateRoute path="/subChannelPerformance" component={SubChannelPerformance} />
    <PrivateRoute path="/subChannelODPerformance" component={SubChannelPerformance} />
    <PrivateRoute path="/segmentation" component={Segmentation} />
    <PrivateRoute path="/subSegmentation" component={SubSegmentation} />
    <PrivateRoute path="/DDSChart" component={DDSChart} />
    <PrivateRoute path="/Interline" component={Interline} />
    <PrivateRoute path="/dataLoadIndicator" component={DataLoadIndicator} />
    <PrivateRoute path="/rpsSelection" component={RPSSelection} />
    <PrivateRoute path="/rpsPos" component={RPSPos} />
    <PrivateRoute path="/rpsRoute" component={RPSRoute} />
    <PrivateRoute path="/rpsSettings" component={RPSSettings} />
    <PrivateRoute path="/forecastAccuracy" component={ForecastAccuracy} />
    <PrivateRoute path="/routeProfitabilitySolution" component={RouteProfitabiltySolutions} />
    <PrivateRoute path="/routeProfitabilityConsolidate" component={RouteProfitabilityConsolidate} />
    <PrivateRoute path="/auditLogs" component={AuditLogs} />
    <PrivateRoute path="/alerts" component={Alerts} />
    <PrivateRoute path="/alertsNew" component={AlertsNew} />
    <PrivateRoute path="/alertDetails" component={AlertDetails} />
    <PrivateRoute path="/alertSetting" component={AlertSetting} />
    <Route path="/targetRational" component={TargetRational} />
    <Route path='/invalidLogin' component={InvalidLogin} />
    <Route path='/404' component={PageNotFound} />
    <Route component={PageNotFound} />

  </div >
);

export default BaseRouter;