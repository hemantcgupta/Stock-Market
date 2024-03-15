import React, { Component } from "react";
import RegionWisePerformance from './regionWisePerformance';
import AnciallaryItems from './AnciallaryItems';
import TopBottomRoutes from './TopBottomRoutes';
import '../RouteDashboard.scss';

class Tables extends Component {

  render() {
    return (
      <div className="x_panel">
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <RegionWisePerformance
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <AnciallaryItems
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            posDashboard={false}
            {...this.props}
          />
        </div>
        <div className="col-md-12 col-sm-4 col-xs-12 table-flex">
          <TopBottomRoutes
            startDate={this.props.startDate}
            endDate={this.props.endDate}
            routeGroup={this.props.routeGroup}
            regionId={this.props.regionId}
            countryId={this.props.countryId}
            routeId={this.props.routeId}
            {...this.props}
          />
        </div>
      </div>
    )
  }
}
export default Tables;